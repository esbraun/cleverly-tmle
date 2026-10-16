r"""Property families of ``structural-nested-mean`` (X44).

Every fit is :func:`tests.studies.canonical_structural_nested_mean.fit`: the declared ``GLM``
nuisances of each setup (a shallow booster on ``N``), the shipped ten outer folds (five on ``N``)
and the setup's blip features.  The setups, their laws and their exact truths are in
:mod:`tests.studies.structural_nested_mean_laws`.

=============================  ===================================================================
family                         cells, replications and size
=============================  ===================================================================
``interval_calibration``       every estimand of the eleven positive setups at n = 2,000, with
                               :data:`CALIBRATION_REPLICATES` per setup, and the two derived
                               controls of each label (a shrunken standard error and added noise)
``root_n_and_efficiency``      one contrast on each of ``S2``, ``H``, ``K`` and ``C``, and a
                               contrast and a blip on the adaptive cell ``N``, at the three sizes
                               of :func:`~tests.studies.canonical_structural_nested_mean.ladder_sizes`,
                               :data:`LADDER_REPLICATES` each
``root_n_rate``                the rate rows of each ladder label
``warning_rate``               the model check's size: a correct model on ``S2`` and ``H`` at
                               n = 1,000 and 2,000 and at two and ten folds, and on the
                               sparse-positivity law ``SP`` at n = 2,000
``power``                      the model check's power on ``LU``, the law whose node-2 blip reads
                               a covariate the structural model omits
``blip_feature_control``       ``LU`` and ``VB`` fitted with too few blip features; a detection
                               rule, so a cell passes when the instrument sees the gap
``unclustered_se``             the SE of ``S2C`` fitted without its cluster column, on the
                               clustered calibration draws; the two level estimands must leave
                               the calibration band from below, the others report
=============================  ===================================================================

**Rules.**  ``interval_calibration`` reads the shared calibration rule: the 99% bootstrap
interval of the SE ratio inside (0.93, 1.07) and the 99% coverage interval inside (0.92, 0.98).
A positive cell also needs its 99% standardized-bias interval inside 0.25 SD
(``bias_equivalent``).  The shared rule reads no bias, and a 0.3 SD bias keeps coverage inside
its band at this R.  A negative control passes when its 99% standardized-bias interval lies beyond
0.25 SD on the side of the computed gap (``bias beyond margin``).  There is no efficiency band: the structural estimator is not the efficient
estimator of the nonparametric model, so :attr:`StudyRecord.calibration_efficiency_ratio` is
``False``, and the noise control's scale is :data:`NOISE_SD`, one pilot sampling SD at the
cell's size.  ``root_n_and_efficiency`` reads the shared ladder rule.  ``warning_rate`` reads the
shared rule: a positive cell passes when the upper end of its exact rejection interval is at most
:math:`\alpha` plus the type-I margin.  ``power`` passes when the lower end is at least 0.80.

Each row of a model-check cell holds the structural minus the nonparametric value of the first
contrast, the standard error of that difference, the population difference as ``truth`` (zero when
the structural model holds), and ``rejected``, the joint Wald test at level :math:`\alpha`.  A
check that reports no test (the rank rule) counts as no rejection.

Every fit's inference status is checked before its row is written (``fit_set_rows``).

**Failures.**  A replicate whose fit raises a recordable error is recorded by
:func:`~tests.studies.evidence.replication.fit_properties` and never redrawn.  It costs each
cell its fit set writes one replicate (:data:`FIT_SET_CELLS`), and the accounting judges each
cell at the declared bound :data:`FAILURE_BOUND` under :data:`PROPERTY_RULES`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.learners import GLM
from cleverly.longitudinal import StructuralNestedMean
from tests import discrete_law_structural as finite
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_structural_nested_mean as study
from tests.studies import structural_nested_mean_laws as laws
from tests.studies.evidence import property_failures as pf
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    se_ratio_leaves_band,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CALIBRATION_N = study.CALIBRATION_N
LADDER = study.LADDER
LADDER_TARGETS = study.LADDER_TARGETS
CALIBRATED = study.CALIBRATED
CALIBRATION_LABELS = study.CALIBRATION_LABELS
LADDER_ESTIMANDS = study.LADDER_ESTIMANDS
LADDER_LABELS = study.LADDER_LABELS
CHECK_CELLS = study.CHECK_CELLS
TARGETS = study.TARGETS
label_of = study.label_of
SIZE_REPLICATES = study.SIZE_REPLICATES
POWER_REPLICATES = study.POWER_REPLICATES
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The replicates of every calibration cell.  The arm-pooled predictive rule asks for the smallest
#: multiple of 400 whose chance that all 61 cells pass is at least 0.95.  The adaptive arm ``N``
#: cannot reach it at any count: its pilot SE ratios spread between cells, and its chance tends to
#: 0.67.  The rule applied to the other arms gives 17,600.  The declared count is above that
#: minimum; the adaptive arm passes with chance 0.45 and the family with 0.44
#: (``tests/unit/test_structural_nested_mean_design.py``).
CALIBRATION_REPLICATES = 21_200
LADDER_REPLICATES = 1_000
#: The empirical SD of each calibration label in the design pilot times :math:`\sqrt{2000}`
#: (``CALIBRATION_N``), so the noise control adds one sampling SD at the cell's size.  It scales
#: the noise control only.  The design test pins these values.
NOISE_SD: dict[str, float] = {
    "s2_blip_a1_l1": 1.3158,
    "s2_blip_a2_l1": 1.1574,
    "s2_ey0": 0.9717,
    "s2_ey_r11": 1.0918,
    "s2_ate_r11": 1.5459,
    "s2_ey_r10": 1.2473,
    "s3_blip_a1_l1": 1.3945,
    "s3_blip_a2_l1": 1.3242,
    "s3_blip_a3_l1": 1.2032,
    "s3_ey0": 0.9960,
    "s3_ate_r111": 1.6515,
    "s3_ate_r101": 1.6592,
    "k_blip_a1_l1": 1.4786,
    "k_blip_a1_l2": 1.5364,
    "k_blip_a2_l1": 1.3067,
    "k_blip_a2_l2": 1.2683,
    "k_ey0": 1.1339,
    "k_ey_r12": 1.4463,
    "k_ate_r12": 1.9280,
    "k_ate_r21": 1.9646,
    "c_blip_a1": 0.8128,
    "c_blip_a2": 0.6931,
    "c_ey0": 0.6959,
    "c_ey_dose_mid": 0.7274,
    "c_ate_dose_mid": 0.4259,
    "c_ate_dose_high": 0.9143,
    "h_blip_a1_l1": 2.0530,
    "h_blip_a1_l1_w": 2.6718,
    "h_blip_a2_l1": 1.6787,
    "h_blip_a2_l1_w": 2.3169,
    "h_blip_a2_l1_a1": 2.2954,
    "h_ey0": 1.0394,
    "h_ey_r11": 1.4078,
    "h_ate_r11": 1.6654,
    "h_ate_r10": 1.7748,
    "p_blip_a1_l1": 1.7918,
    "p_blip_a2_l1": 1.4875,
    "p_blip_a2_l1_a1": 2.2981,
    "p_ey0": 1.0318,
    "p_ate_r11": 1.5542,
    "p_ey_r01": 1.2680,
    "m_blip_a1_l1": 1.3032,
    "m_blip_a2_l1": 1.3563,
    "m_ey0": 1.3374,
    "m_ate_r11": 1.7255,
    "l_blip_a1_l1": 1.4007,
    "l_blip_a2_l1": 1.5949,
    "l_blip_a2_l1_s2": 2.2809,
    "l_ey0": 0.9706,
    "n_blip_a1_l1": 1.4031,
    "n_blip_a2_l1": 1.2001,
    "n_ey0": 1.3383,
    "n_ate_r11": 1.7325,
    "s2w_blip_a1_l1": 1.5830,
    "s2w_ey0": 1.1960,
    "s2w_ey_r11": 1.2541,
    "s2w_ate_r11": 1.8567,
    "s2c_blip_a1_l1": 2.5890,
    "s2c_ey0": 3.5550,
    "s2c_ey_r11": 3.6877,
    "s2c_ate_r11": 3.1175,
}

# ------------------------------------------------------------------------- model check


@dataclass(frozen=True)
class FitSet:
    """One set of draws and the fits it feeds.

    Parameters
    ----------
    kind : str
        ``"calibration"`` (all estimands of a setup), ``"ladder"`` (one contrast at one size),
        ``"size"`` or ``"power"`` (the model check).
    setup : str
        The setup whose law is drawn.
    n : int
        The sample size.
    replicates : int
        The declared replicates.
    folds : int or None
        The outer folds of a model-check fit; ``None`` for the setup's own.
    """

    kind: Literal["calibration", "ladder", "size", "power", "control", "positivity"]
    setup: str
    n: int
    replicates: int
    folds: int | None = None

    @property
    def property(self) -> str:
        """The family this set's rows are filed under."""
        return {
            "calibration": "interval_calibration",
            "ladder": "root_n_and_efficiency",
            "size": "warning_rate",
            "power": "power",
            "control": "blip_feature_control",
            "positivity": "positivity_reporting",
        }[self.kind]


def _fit_sets() -> dict[str, FitSet]:
    out: dict[str, FitSet] = {}
    for setup in CALIBRATED:
        out[f"calibration_{setup.lower()}"] = FitSet(
            "calibration", setup, CALIBRATION_N, CALIBRATION_REPLICATES
        )
    for setup in LADDER_ESTIMANDS:
        for size in study.ladder_sizes(setup):
            out[f"ladder_{setup.lower()}_n_{size}"] = FitSet(
                "ladder", setup, size, LADDER_REPLICATES
            )
    for setup in study.NEGATIVE_CONTROLS:
        out[f"control_{setup.lower()}"] = FitSet(
            "control", setup, CALIBRATION_N, study.CONTROL_REPLICATES
        )
    out["positivity_sp"] = FitSet(
        "positivity", "SP", study.POSITIVITY_N, study.POSITIVITY_REPLICATES
    )
    for name, (kind, setup, n, folds) in CHECK_CELLS.items():
        replicates = study.check_replicates(name)
        out[f"check_{name}"] = FitSet(kind, setup, n, replicates, folds)  # type: ignore[arg-type]
    return out


#: Every fit set: ``name -> FitSet``.  One draw of the law per replicate.
FIT_SETS: dict[str, FitSet] = _fit_sets()


def sample_seed(fit_set: str, replicate: int) -> int:
    """The seed of one draw: this study's own ``property_sample`` stream."""
    return stream_seed(STUDY, "property_sample", fit_set, replicate)


def draw(fit_set: str, replicate: int) -> pd.DataFrame:
    """One draw of a fit set's law."""
    spec = FIT_SETS[fit_set]
    return laws.sample(spec.setup, spec.n, sample_seed(fit_set, replicate))


@functools.cache
def population_limit(name: str) -> dict[str, float]:
    """The limit of the declared fit on a setup, from the exact table at population weights.

    A saturated ``GLM`` returns each nuisance as the population conditional mean, so the fit
    returns the population solution.  On a setup whose structural model holds this is the truth.
    On the negative controls it is the (biased) limit.
    """
    chosen = laws.setup(name)
    law = chosen.law
    table = finite.frame(law).drop(columns="E")
    saturated = len(law.baseline) + len(law.treatments) * 2
    estimator = StructuralNestedMean(
        chosen.regimens or None,
        reference=chosen.reference or None,
        blip_features=chosen.features,
        outcome_learner=GLM(family="gaussian", interactions=saturated, tol=1e-13),
        treatment_learner=GLM(interactions=saturated, tol=1e-13),
        n_folds=1,
        simultaneous=False,
    )
    result = estimator.fit(
        table,
        outcome="Y",
        treatment=law.treatments,
        baseline=law.baseline,
        time_varying=law.time_varying,
        weights="p",
        family="gaussian",
        continuous_treatment=list(law.doses),
    )
    return {key: float(result[key].psi) for key in result}


def check_names(name: str) -> tuple[str, ...]:
    """The contrasts a model check of setup ``name`` compares, against its reference."""
    chosen = laws.setup(name)
    return tuple(
        f"ate_regimen[{label} vs {chosen.reference}]"
        for label in chosen.regimens
        if label != chosen.reference
    )


@functools.cache
def check_gaps(name: str) -> dict[str, float]:
    """The population structural minus nonparametric value of each contrast.

    Zero when the structural model holds.  The nonparametric contrast is the g-formula contrast,
    so on ``LU`` the gap is the structural limit minus the regimen contrast.
    """
    limit = population_limit(name)
    return {
        contrast: float(limit[contrast] - laws.truth(name, contrast))
        for contrast in check_names(name)
    }


@functools.cache
def check_contrast(name: str) -> str:
    """The contrast a model-check row reports: the one with the largest population gap.

    The first contrast when every gap is zero (the structural model holds).
    """
    gaps = check_gaps(name)
    first = next(iter(gaps))
    best = max(gaps, key=lambda contrast: abs(gaps[contrast]))
    return best if abs(gaps[best]) > 1e-9 else first


def check_truth(name: str) -> float:
    """The population gap of the contrast :func:`check_contrast` names."""
    return check_gaps(name)[check_contrast(name)]


# --------------------------------------------------------------------------- the rows


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


def _check_row(name: str, spec: FitSet, replicate: int, check: Any) -> dict[str, Any]:
    truth = check_truth(spec.setup)
    index = check.names.index(check_contrast(spec.setup))
    gap = float(check.difference[index])
    error = float(np.sqrt(check.covariance[index, index]))
    return {
        "property": spec.property,
        "cell": f"{name}__model_check",
        "role": "positive",
        "replicate": replicate,
        "n": spec.n,
        "requested_replicates": spec.replicates,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": gap,
        "std_error": error,
        "covered": int(abs(gap - truth) <= CRITICAL * error),
        "rejected": int(bool(check.tested and check.pvalue < STUDY.margins.alpha)),
    }


def fit_set_rows(fit_set: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes.

    Each fit's inference status is checked before a row is written
    (:func:`~tests.studies.canonical_structural_nested_mean.require_status`): a status other than
    ``influence_curve`` stops the run rather than calibrating the wrong curve.
    """
    spec = FIT_SETS[fit_set]
    rows: list[dict[str, Any]] = []
    if spec.kind == "ladder":
        estimands = LADDER_ESTIMANDS[spec.setup]
        result = study.fit(spec.setup, frame)
        study.require_status(result, estimands)
        for estimand in estimands:
            rows.append(
                _row(
                    "root_n_and_efficiency",
                    f"{label_of(spec.setup, estimand)}__n_{spec.n}",
                    "positive",
                    replicate,
                    spec.n,
                    spec.replicates,
                    laws.truth(spec.setup, estimand),
                    result[estimand],
                )
            )
        return rows
    if spec.kind in ("size", "power"):
        name = fit_set.removeprefix("check_")
        result = study.fit(spec.setup, frame, folds=spec.folds, order=study.CHECK_INTERACTIONS)
        study.require_status(result, [])
        check = result.check_model(q_bounds=laws.outcome_support(spec.setup))
        return [_check_row(name, spec, replicate, check)]
    result = study.fit(spec.setup, frame)
    names = list(laws.setup(spec.setup).estimands)
    study.require_status(result, names)
    if spec.kind == "positivity":
        return [
            _row(
                "positivity_reporting",
                f"{label_of(spec.setup, estimand)}__weak_positivity",
                DIAGNOSTIC_ROLE,
                replicate,
                spec.n,
                spec.replicates,
                laws.truth(spec.setup, estimand),
                result[estimand],
            )
            for estimand in names
        ]
    if spec.kind == "control":
        for estimand in names:
            rows.append(
                _row(
                    "blip_feature_control",
                    f"{label_of(spec.setup, estimand)}__too_few_features",
                    "control",
                    replicate,
                    spec.n,
                    spec.replicates,
                    laws.truth(spec.setup, estimand),
                    result[estimand],
                )
            )
        return rows
    for estimand in names:
        label = label_of(spec.setup, estimand)
        rows.append(
            _row(
                "interval_calibration",
                f"{label}__correctly_specified",
                "positive",
                replicate,
                spec.n,
                spec.replicates,
                laws.truth(spec.setup, estimand),
                result[estimand],
            )
        )
    if spec.setup == "S2C":
        unclustered = study.fit(spec.setup, frame, clustered=False)
        study.require_status(unclustered, [])
        for estimand in names:
            label = label_of(spec.setup, estimand)
            rows.append(
                _row(
                    "unclustered_se",
                    f"{label}__unclustered",
                    DIAGNOSTIC_ROLE,
                    replicate,
                    spec.n,
                    spec.replicates,
                    laws.truth(spec.setup, estimand),
                    unclustered[estimand],
                )
            )
    return rows


# ------------------------------------------------------------------------------ the grid


@dataclass(frozen=True)
class DeclaredLaw:
    """The law a declared cell reads, by name, with its exact truth of every estimand.

    Parameters
    ----------
    name : str
        The law's name.  A diagnostic family that reads the fits of another family carries
        the family in its name, so the collision gate keys its stream apart from the claim.
    setup : str
        The setup whose truths this law publishes.
    """

    name: str
    setup: str

    def truth(self) -> dict[str, float]:
        """Return ``estimand -> truth`` for every estimand of this law."""
        out = dict(laws.truths(self.setup))
        if laws.setup(self.setup).regimens:
            out["model_check"] = check_truth(self.setup)
        return out


#: The families that report a diagnostic on the fits of a claim family.
SHARED_STREAM_FAMILIES = frozenset({"unclustered_se"})


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its law, estimand, size and stream root."""
    return tuple(cell for cell, _ in _cells_and_fit_sets())


def _cells_and_fit_sets() -> list[tuple[PropertyCell, str]]:
    """Every sampled cell beside the fit set whose draws it reads."""
    cells: list[tuple[PropertyCell, str]] = []

    def add(family: str, cell: str, role: str, fit_set: str, estimand: str) -> None:
        spec = FIT_SETS[fit_set]
        name = spec.setup if family not in SHARED_STREAM_FAMILIES else f"{spec.setup}_{family}"
        cells.append(
            (
                PropertyCell(
                    property=family,
                    cell=cell,
                    dgp=DeclaredLaw(name, spec.setup),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=spec.n,
                    replicates=spec.replicates,
                    seed=sample_seed(fit_set, 0),
                    role=role,
                    estimand=estimand,
                ),
                fit_set,
            )
        )

    for label, (setup, estimand) in TARGETS.items():
        for kind, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{kind}",
                role,
                f"calibration_{setup.lower()}",
                estimand,
            )
    for label, (setup, estimand) in LADDER_TARGETS.items():
        for size in study.ladder_sizes(setup):
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                "positive",
                f"ladder_{setup.lower()}_n_{size}",
                estimand,
            )
    for name, (kind, *_) in CHECK_CELLS.items():
        family = "warning_rate" if kind == "size" else "power"
        add(family, f"{name}__model_check", "positive", f"check_{name}", "model_check")
    for label, (setup, estimand) in study.CONTROL_TARGETS.items():
        add(
            "blip_feature_control",
            f"{label}__too_few_features",
            "control",
            f"control_{setup.lower()}",
            estimand,
        )
    for label, (_, estimand) in study.POSITIVITY_TARGETS.items():
        add(
            "positivity_reporting",
            f"{label}__weak_positivity",
            DIAGNOSTIC_ROLE,
            "positivity_sp",
            estimand,
        )
    for estimand in laws.setup("S2C").estimands:
        label = label_of("S2C", estimand)
        add("unclustered_se", f"{label}__unclustered", DIAGNOSTIC_ROLE, "calibration_s2c", estimand)
    return cells


# ------------------------------------------------------------------------------- the loop


def _draw_key(key: tuple[str, int]) -> pd.DataFrame:
    return draw(*key)


def _fit_key(key: tuple[str, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_set_rows(key[0], key[1], frame)


#: Each calibration positive arm to the controls derived from it.
DERIVED_CELLS: dict[str, tuple[str, ...]] = calibration_derived(CALIBRATION_LABELS)


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
    declared = FIT_SETS[fit_set].replicates
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
    present = set(rows["cell"])
    labels = [label for label in CALIBRATION_LABELS if f"{label}__correctly_specified" in present]
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

#: The ``S2C`` estimands whose SE, fitted without the cluster column, must leave the calibration
#: band from below.  The two level estimands carry the cluster intercept, so ignoring the
#: clusters understates their SE by half.  The blip and the contrast difference the intercept
#: away, and their unclustered SE reads inside the band, so those cells report.
GATED_ESTIMANDS = ("ey_reference", "ey_regimen[r11]")
#: The cells that must leave the calibration band from below: the unclustered SE of ``S2C``.
GATED_DIRECTIONS: dict[str, Literal["below", "above"]] = {
    f"{label_of('S2C', estimand)}__unclustered": "below" for estimand in GATED_ESTIMANDS
}


def _own_seed(ctx: pf.Context, row: pd.Series) -> int:
    return stream_seed(ctx.record, str(row["property"]), str(row["cell"]))


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


@functools.cache
def control_gap(label: str) -> float:
    """The population limit of a negative control's fit minus its g-formula truth."""
    setup, estimand = study.CONTROL_TARGETS[label]
    return population_limit(setup)[estimand] - laws.truth(setup, estimand)


def _control_side(ctx: pf.Context, row: pd.Series) -> int:
    """The side of the truth on which a control's limit lies: the sign of its computed gap."""
    return 1 if control_gap(cell_label(str(row["cell"]))) > 0 else -1


def _control_rule() -> pf.FamilyRule:
    """The negative controls: a detection rule.

    A control passes when its 99% standardized-bias interval lies beyond the margin on the side
    of the computed gap, as every other control of the instrument passes by leaving its band.
    The SE ratio and the coverage are published and gate nothing.
    """
    return pf.FamilyRule(
        families=frozenset({"blip_feature_control"}),
        shared="summarize_properties",
        clauses=(
            pf.bias_clause(
                "bias beyond margin on the side of the gap",
                kind="beyond",
                side=_control_side,
                applies=pf.every,
            ),
        ),
    )


_SIDES: tuple[Literal["above", "below"], ...] = ("above", "below")


def _direction_rule() -> pf.FamilyRule:
    """A gated cell's SE-ratio interval leaves the band on its predicted side."""

    def leaves(side: Literal["below", "above"]) -> pf.Judge:
        def judge(ctx: pf.Context, row: pd.Series, iv: Any) -> pf.Judged:
            low, high = ctx.margins.calibration_se_ratio
            slack = low - iv.high if side == "below" else iv.low - high
            return pf.Judged(slack, se_ratio_leaves_band(iv, (low, high), side))

        return judge

    return pf.FamilyRule(
        families=frozenset({"unclustered_se"}),
        shared="summarize_properties",
        clauses=tuple(
            pf.se_interval_clause(
                f"se ratio {side} band",
                which="se_ratio",
                judge=leaves(side),
                seed=_own_seed,
                applies=pf.cells(*(cell for cell, s in GATED_DIRECTIONS.items() if s == side)),
            )
            for side in _SIDES
        ),
    )


#: The failure accounting's declared rules.
PROPERTY_RULES = (_calibration_rule(), _control_rule(), _direction_rule())


def _own_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    margins = STUDY.margins
    for family in ("blip_feature_control", "unclustered_se"):
        for index in summary.index[summary["property"] == family]:
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
            if family == "blip_feature_control":
                margin = float(summary.loc[index, "bias_margin"])
                low = float(summary.loc[index, "bias_ci_lower"])
                high = float(summary.loc[index, "bias_ci_upper"])
                beyond = low > margin if control_gap(cell_label(cell)) > 0 else high < -margin
                summary.loc[index, "passed"] = bool(beyond)
                continue
            side = GATED_DIRECTIONS.get(cell)
            # A gated cell must leave the calibration band on its predicted side, so it shows
            # that the calibration instrument would catch an ignored cluster column.
            summary.loc[index, "passed"] = (
                True
                if side is None
                else se_ratio_leaves_band(ratio, margins.calibration_se_ratio, side)
            )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    labels = tuple(
        label
        for label, (setup, _) in LADDER_TARGETS.items()
        if (rows["cell"] == f"{label}__n_{study.ladder_sizes(setup)[0]}").any()
    )
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=labels)
    calibration_verdicts(summary, margins=STUDY.margins)
    # The shared calibration rule reads the SE ratio and coverage only, and a bias of 0.3 SD
    # keeps coverage inside its band at this R.  Every positive calibration cell also needs
    # its 99% bias interval inside 0.25 SD.
    positive = (summary["property"] == "interval_calibration") & summary["cell"].str.endswith(
        "__correctly_specified"
    )
    summary.loc[positive, "passed"] = summary.loc[positive, "passed"].astype(bool) & summary.loc[
        positive, "bias_equivalent"
    ].astype(bool)
    # A reported family states no verdict.
    summary.loc[summary["property"] == "positivity_reporting", "passed"] = True
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
