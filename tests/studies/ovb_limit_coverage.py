"""Registered evidence for the confidence limits of the part G omitted-variable bounds (X82 SG).

X82 part G extended the omitted-variable bound to a response mechanism, an intermediate
variable, the regime and MSM axes, and a shift on declared dose atoms.  Each extension's
one-sided confidence limit stays descriptive until a registered cell measures its coverage
(``X82-SG-plan.md`` in the plans repository).  This study holds one cell per extension:

====  ===================  ====================================================  ===========
cell  ``ovb_limit`` rule   law (:mod:`tests.studies.sensitivity_limit_laws`)      parameters
====  ===================  ====================================================  ===========
O1    ``response``         the response version, ``Y`` missing at random          ``ate``, ``ey1``
O2    ``intermediate``     the mediated version                                  the CDE at ``Z = 0`` and ``Z = 1``
O3    ``regime``           the complete version, with a stochastic regime         ``ey_regime[coin]``, ``ate_regime[coin vs always]``
O4    ``msm``              the complete version, ``MSM.linear(modifiers=("W2",))``  ``msm[W2]``, ``msm[a:W2]``
O5    ``shift``            :mod:`tests.discrete_law_shift`, on its dose atoms      ``ey_policy[+1]``, ``ate_policy[+1 vs natural course]``
====  ===================  ====================================================  ===========

As in ``omitted-variable-bound-se``, each row reads one end of a bound as an estimate of its
population value, and recovers the standard error from the reported one-sided limit.  Every
truth is a closed form of the law's tables.  Every nuisance is a saturated logistic model, so
each fit is correctly specified.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.estimators import TMLE
from cleverly.interventions import Shift, Static, Stochastic
from cleverly.msm import MSM
from cleverly.sensitivity import omitted_variable as omitted_variable_module
from cleverly.validation._interpretation import fit_configuration
from tests import discrete_law_shift as shift_law
from tests.parallel import STUDY_JOBS
from tests.studies import sensitivity_limit_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

PRIMARY_REPLICATES = 1_600
PRIMARY_N = 2_000
#: 20262301 was ``canonical-composite-drtmle``'s seed, so the first run shared its resampling
#: streams.  The study was rerun on fresh seeds and its results accepted as drawn.
SEED = 20262321
#: 20262322 was also ``marginal-bound-limit-coverage``'s resampling seed, declared first, so
#: registering that study shared every resampling stream.  Only this entropy moved: the primary
#: rows still draw from ``SEED``.  The property samples draw from this entropy, so the study was
#: rerun and its results accepted as drawn.
RESAMPLING_SEED = 20262325

#: The two-sided 95% multiplier of each row's interval, and the bound's one-sided multiplier.
TWO_SIDED = float(stats.norm.ppf(0.975))
ONE_SIDED = float(stats.norm.ppf(0.95))

ENDS = ("lower", "upper")
COVARIATES = ("W1", "W2")

#: ``g*(a)`` of the stochastic regime of O3, for ``a = 0`` and ``a = 1``.
COIN = (0.3, 0.7)


@dataclass(frozen=True)
class Cell:
    """One declared cell.

    Parameters
    ----------
    rule : str
        The ``ovb_limit`` rule the cell's limits resolve through.
    parameters : mapping of str to tuple
        Each row label, and the package parameter it reads: ``(name,)``, or
        ``(name, level)`` for a CDE at an intermediate level.
    cf_y, cf_d : float
        The declared strength.  ``rho`` is one.
    """

    rule: str
    parameters: Mapping[str, tuple[Any, ...]]
    cf_y: float = 0.3
    cf_d: float = 0.2

    @property
    def strength(self) -> float:
        """``|rho| sqrt(cf_y cf_d / (1 - cf_d))``."""
        return math.sqrt(self.cf_y * self.cf_d / (1.0 - self.cf_d))


CELLS: dict[str, Cell] = {
    "O1": Cell("response", {"ate": ("ate",), "ey1": ("ey1",)}),
    "O2": Cell("intermediate", {"cde_z0": ("ate", 0.0), "cde_z1": ("ate", 1.0)}),
    "O3": Cell(
        "regime",
        {"coin": ("ey_regime[coin]",), "coin_vs_always": ("ate_regime[coin vs always]",)},
    ),
    "O4": Cell("msm", {"w2": ("msm[W2]",), "a_w2": ("msm[a:W2]",)}),
    "O5": Cell(
        "shift",
        {
            "plus1": ("ey_policy[+1]",),
            "plus1_vs_natural": ("ate_policy[+1 vs natural course]",),
        },
    ),
}

SCENARIOS = {
    cell: tuple(f"{label}_{end}" for label in spec.parameters for end in ENDS)
    for cell, spec in CELLS.items()
}

#: The file that records the fit configuration of each cell.
CELLS_FILE = "cells.csv"

PROPERTY_CELLS = {
    "interval_calibration": (
        *(f"{cell}_{row}__correctly_specified" for cell, rows in SCENARIOS.items() for row in rows),
        *(f"{cell}_{row}__inflated_se_control" for cell, rows in SCENARIOS.items() for row in rows),
    ),
    "response_mask_necessity": tuple(
        f"O1_{row}__{arm}" for row in SCENARIOS["O1"] for arm in ("masked", "unmasked_sigma2")
    ),
}

STUDY = StudyRecord(
    name="omitted-variable bound limit coverage",
    slug="ovb-limit-coverage",
    artifacts=ROOT / "tests" / "canonical" / "ovb_limit_coverage",
    document="docs/technical-reference/method-evidence/ovb-limit-coverage.md",
    anchor="omitted-variable-bound-limit-coverage",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-ovb-limit",
    reference=None,
    failure_bound={"cleverly-ovb-limit": 0.01},
    modules=(
        "tests/studies/ovb_limit_coverage.py",
        "tests/studies/ovb_limit_coverage_properties.py",
        "tests/studies/sensitivity_limit_laws.py",
        "tests/discrete_law_shift.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.ovb_limit_coverage",
    properties_module="tests.studies.ovb_limit_coverage_properties",
    property_artifacts=(CELLS_FILE,),
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

CONFIGURATION = {
    "laws": "tests.studies.sensitivity_limit_laws: the part G law in its response, mediated "
    "and complete versions, and tests.discrete_law_shift on its four dose atoms",
    "fit": "in-sample TMLE, simultaneous=False, g_bounds=(0.01, 0.99); every nuisance a "
    "logistic regression on all pairwise products of its inputs (C=1e8), saturated in the "
    "cells it reads; W enters as the indicators W1 and W2",
    "nu2_estimator": "auto, which resolves to doubly_robust",
    "strength": {
        cell: {"cf_y": spec.cf_y, "cf_d": spec.cf_d, "rho": 1.0} for cell, spec in CELLS.items()
    },
    "primary_row": "estimate = one end of the bound; std_error = (lower - ci_lower) / z_0.95 "
    "for a lower end and (ci_upper - upper) / z_0.95 for an upper end; interval = estimate "
    "+/- 1.959964 std_error",
    "truth": "closed forms of the law tables (sensitivity_limit_laws)",
}


# ------------------------------------------------------------------------------- the laws


def _frame(cell: str) -> pd.DataFrame:
    if cell == "O1":
        return laws.exact_frame("response")
    if cell == "O2":
        return laws.exact_frame("mediated")
    if cell in ("O3", "O4"):
        return laws.exact_frame("complete")
    if cell == "O5":
        return laws.shift_exact_frame()
    raise KeyError(f"{STUDY.slug} has no scenario {cell!r}")


def _regime_tables() -> dict[str, np.ndarray]:
    """The weight table ``c(a, w)``, indexed ``[w, a]``, of each regime parameter of O3."""
    coin = np.tile(np.asarray(COIN), (3, 1))
    always = np.tile([0.0, 1.0], (3, 1))
    return {"coin": coin, "coin_vs_always": coin - always}


def _msm_design(a: int, w: int) -> list[float]:
    """``phi(a, w)`` of ``MSM.linear(modifiers=("W2",))``: intercept, ``a``, ``W2``, ``a W2``."""
    w2 = float(w == 2)
    return [1.0, float(a), w2, float(a) * w2]


#: The index of each O4 coefficient in :func:`_msm_design`.
_MSM_INDEX = {"w2": 2, "a_w2": 3}


def population_terms(cell: str, label: str) -> tuple[float, float, float]:
    """``(psi0, sigma0^2, nu0^2)`` of one parameter, from the law's tables."""
    if cell == "O1":
        psi = laws.arm_mean(1) - laws.arm_mean(0) if label == "ate" else laws.arm_mean(1)
        return psi, laws.sigma2("response"), laws.response_nu2(label)
    if cell == "O2":
        z = int(CELLS["O2"].parameters[label][1])
        return laws.mediated_psi(z), laws.mediated_sigma2(z), laws.mediated_nu2(z)
    if cell == "O3":
        table = _regime_tables()[label]
        return laws.regime_psi(table), laws.sigma2("complete"), laws.regime_nu2(table)
    if cell == "O4":
        beta, _, _ = laws.msm_terms(_msm_design)
        index = _MSM_INDEX[label]
        return float(beta[index]), laws.sigma2("complete"), laws.msm_nu2(_msm_design, index)
    if cell == "O5":
        psi, s2, nu2 = laws.shift_terms("+1")
        if label == "plus1":
            return psi, s2, nu2
        natural, _, _ = laws.shift_terms("natural course")
        # The contrast's representer is alpha_{+1} - 1, whose second moment is nu2 - 1.
        return psi - natural, s2, nu2 - 1.0
    raise KeyError(cell)


def truths(cell: str) -> dict[str, float]:
    """The population value of each end of each bound of ``cell``."""
    spec = CELLS[cell]
    values: dict[str, float] = {}
    for label in spec.parameters:
        psi, s2, nu2 = population_terms(cell, label)
        bias = spec.strength * math.sqrt(s2 * nu2)
        values[f"{label}_lower"] = psi - bias
        values[f"{label}_upper"] = psi + bias
    return values


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample from an explicit seed for the published-seed audit."""
    frame = laws.sample(_frame(scenario), n, np.random.default_rng(seed))
    return frame, truths(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# --------------------------------------------------------------------------------- the fit


def _estimator(**settings: Any) -> TMLE:
    return TMLE(
        outcome_learner=laws.saturated_classifier(),
        treatment_learner=laws.saturated_classifier(),
        missingness_learner=laws.saturated_classifier(),
        intermediate_learner=laws.saturated_classifier(),
        cross_fit=False,
        simultaneous=False,
        g_bounds=(0.01, 0.99),
        random_state=0,
        **settings,
    )


def fit(cell: str, frame: pd.DataFrame, **data: Any) -> Any:
    """The declared fit of ``cell``.  ``data`` reaches ``fit`` (the design test's weights)."""
    common = {"outcome": "Y", "treatment": "A", "covariates": list(COVARIATES), **data}
    if cell == "O1":
        return _estimator(estimands=("ey1", "ate")).fit(frame, delta="D", **common).single()
    if cell == "O2":
        return _estimator(estimands=("ate",)).fit(frame, intermediate="Z", **common)
    if cell == "O3":
        regimes = [
            Static(1, name="always"),
            Static(0, name="never"),
            Stochastic(
                lambda w: np.tile(list(COIN), (len(w), 1)), name="coin", density_kind="known"
            ),
        ]
        return _estimator(interventions=regimes).fit(frame, **common).single()
    if cell == "O4":
        return _estimator(msm=MSM.linear(modifiers=("W2",))).fit(frame, **common).single()
    if cell == "O5":
        policies = [
            Shift(0.0, cap=shift_law.CAP, name="natural course"),
            Shift(shift_law.DELTA, cap=shift_law.CAP, name="+1"),
        ]
        return (
            _estimator(policies=policies, density_bins=len(shift_law.DOSES))
            .fit(frame, dose_atoms=shift_law.DOSES, **common)
            .single()
        )
    raise KeyError(cell)


def parameter_result(cell: str, fitted: Any, label: str) -> tuple[Any, str]:
    """The result and the package name that row ``label`` of ``cell`` reads."""
    reference = CELLS[cell].parameters[label]
    if cell == "O2":
        return fitted[reference[1]], reference[0]
    return fitted, reference[0]


def bound_values(cell: str, fitted: Any, label: str) -> dict[str, tuple[float, float]]:
    """Each end of one bound and its standard error, read back from the reported limit."""
    spec = CELLS[cell]
    result, name = parameter_result(cell, fitted, label)
    bounds = omitted_variable_module.omitted_variable_bounds(
        result, name, cf_y=spec.cf_y, cf_d=spec.cf_d, rho=1.0
    )
    if bounds.features != (spec.rule,):
        raise RuntimeError(
            f"{cell} {label}: the bound's features {bounds.features} are not ({spec.rule!r},)"
        )
    return {
        "lower": (bounds.lower, (bounds.lower - bounds.ci_lower) / ONE_SIDED),
        "upper": (bounds.upper, (bounds.ci_upper - bounds.upper) / ONE_SIDED),
    }


def _row(
    scenario: str, replicate: int, n: int, estimand: str, truth: float, estimate: float, se: float
) -> dict[str, Any]:
    low = estimate - TWO_SIDED * se
    high = estimate + TWO_SIDED * se
    return {
        "implementation": STUDY.implementation,
        "scenario": scenario,
        "replicate": replicate,
        "n": n,
        "estimand": estimand,
        "truth": float(truth),
        "estimate": float(estimate),
        "inference_estimate": float(estimate),
        "std_error": float(se),
        "ci_lower": float(low),
        "ci_upper": float(high),
        "inference_scale": "identity",
        "covered": int(low <= truth <= high),
        "initial_estimate": math.nan,
    }


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema, one row per bound end."""
    fitted = fit(scenario, frame)
    rows: list[dict[str, Any]] = []
    for label in CELLS[scenario].parameters:
        values = bound_values(scenario, fitted, label)
        for end in ENDS:
            estimand = f"{label}_{end}"
            estimate, se = values[end]
            rows.append(
                _row(scenario, replicate, len(frame), estimand, truth[estimand], estimate, se)
            )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    frame, truth = drawn
    return cleverly_rows(frame, truth, scenario, index)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


def cell_configurations() -> pd.DataFrame:
    """The fit configuration of each row, read off one fit per cell.

    The draw comes from a stream of its own.  Every learner is fixed and fits in sample, so
    one draw records the configuration that every replication of the cell fits.
    """
    records = []
    for cell in CELLS:
        frame, _ = draw_from_seed(cell, PRIMARY_N, stream_seed(STUDY, "configuration", cell))
        fitted = fit(cell, frame)
        for label in CELLS[cell].parameters:
            result, name = parameter_result(cell, fitted, label)
            configuration = asdict(fit_configuration(result, name))
            configuration["learners"] = " | ".join(configuration["learners"])
            records.append({"scenario": cell, "parameter": label, **configuration})
    return pd.DataFrame(records)
