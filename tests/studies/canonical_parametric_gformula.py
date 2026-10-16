r"""Registered evidence for the parametric g-formula (``parametric-gformula``, X45).

The subject is :class:`~cleverly.longitudinal.ParametricGFormula` (NICE), a forward simulation
of node-specific main-terms models.  The laws and their exact truths are in
:mod:`tests.studies.parametric_gformula_laws`, and the property families in
:mod:`tests.studies.parametric_gformula_properties`.

This study is Python only, and its record declares no comparator.  ``pygformula`` 1.2.0 fits
one pooled model per covariate over time, so on these laws it fits a different model.  It is
compared on construction-matched fixtures instead (:mod:`tests.studies.parametric_gformula_fixtures`,
run by ``tests/canonical/parametric_gformula_study/compare_pygformula.py`` in its own image).

**Laws.**  Every covariate law is exactly the main-terms model the estimator fits, in the
estimator's full-history design.  Treatment and censoring are main-terms logistic in the
history.  So the estimator is correctly specified on every law but ``G4Q``.

=======  ==========================================================================================
law      structure and plans
=======  ==========================================================================================
``G1``   :math:`T = 3`, a binary :math:`L_t`, a continuous :math:`W`, censoring and a binary
         end-of-study outcome.  Plans: ``treat_all``, ``treat_none``, the dynamic ``rule``
         :math:`A_t = L_t`, and the odds tilt ``odds x2``
``G2``   :math:`T = 3`, a Gaussian then a binary covariate per node with a steep slope, a
         continuous outcome.  Plans: ``treat_all``, ``treat_none`` and the known stochastic
         ``policy_b``
``G3``   :math:`T = 4`, a binary :math:`L_t`, censoring, two competing causes drawn from a
         baseline-category multinomial.  Plans: ``treat_all_4`` and ``treat_none_4``, at the
         horizons 2 and 4
``G4``   :math:`T = 3`, a Gaussian covariate and a continuous dose per node, a continuous outcome.
         Plans: the additive shift ``shift_half`` against ``natural course``, the fixed dose
         ``dose_half``, and the rule-set dose ``rule_dose`` :math:`A_t = 0.5 + 0.5 L_t`
``G4Q``  ``G4`` with a quadratic dose term in the outcome.  A main-terms fit extrapolates; the
         bias is reported
``G5``   :math:`T = 3`, a binary then a three-level categorical covariate per node, a binary
         treatment and a continuous outcome with a small residual SD.  Plans: ``treat_all``,
         ``treat_none`` and the odds tilt ``odds x2``.  The law lifts the binary-covariate,
         categorical-covariate and tilt terms of the influence curve
=======  ==========================================================================================

**Primary scenarios.**  One scenario per law at :data:`PRIMARY_N` with
:data:`PRIMARY_REPLICATES` replicates.  The shared independent-performance tests read each
row: the standardized bias inside 0.25 SD, the coverage floor 0.90 and the SE sanity band.

**Streams.**  The sampling stream of a replicate comes from :func:`replicate_seed`.  The
estimator's ``random_state`` comes from the study's own ``primary_simulation`` stream, keyed by
scenario and replicate, so the simulation error is independent of the sample and across
replicates.  A fixed seed in every replicate would correlate that error across the whole study.

**Red-cell reading** (declared before any run).  The study publishes under ``reporting``.  The
declared red set is empty: no cell is expected to fail.  The reported cells (the ``G1`` ICE
pairs, the shared-target ``LTMLE`` and ``IPSI`` fits, ``G4Q``, and every
``mc_removed`` cell) carry no verdict.  Each gated calibration cell is sized for a pass chance of
at least 0.95 at its shrunk SE ratio, and at least 0.925 with the posterior SD of that ratio
(about 0.0085) counted, above the floor of 0.90.  Some finite-sample reds are therefore expected
even for a perfect estimator.  The design test computes the count over the :data:`GATED_CELLS`
gated cells, at the empirical-Bayes shrunk SE ratio of each cell (the ratios are near, not at,
1), with the 42 gated derived-control cells counted: :data:`EXPECTED_RED_COUNT` = 1.26 reds, and
a chance of :data:`PROBABILITY_OF_A_RED` = 0.72 of at least one.  Two reds read as expected.
A red is therefore not a surprise.  No replicate count rises to hide it.

A red row is a defect suspect.  It is a finite-sample red only when all three hold: its 99%
SE-ratio interval contains 1, its coverage is consistent with 0.95, and the diagnosis finds no
defect.  It then goes to ``reporting`` with an owner, declared before any repeat run.  A defect
is fixed and the study re-runs under a fresh declaration.  A replicate that raises a recordable
error is recorded and never redrawn; the declared failure bound is :data:`FAILURE_BOUND` per
cell.  No budget, margin, law, learner or seed moves after a verdict is seen.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from cleverly.interventions import Incremental, Shift, Stochastic
from cleverly.learners import GLM
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    DynamicRegimen,
    ParametricGFormula,
    SequentialGComputation,
)
from tests.parallel import STUDY_JOBS
from tests.studies import parametric_gformula_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SEED = 20264501
RESAMPLING_SEED = 2026450101

#: The three sizes of the ladder.
LADDER = (2_000, 4_000, 8_000)
#: The size of every calibration cell.
CALIBRATION_N = 2_000
PRIMARY_N = LADDER[-1]
PRIMARY_REPLICATES = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01

IMPLEMENTATION = "cleverly-parametric-gformula"
#: The ``pygformula`` version the fixture comparison pins, and the tag of its image.
PYGFORMULA = "1.2.0"

#: The expected count of finite-sample reds under perfect calibration, over :data:`GATED_CELLS`
#: gated cells (72 measured cells and 42 derived controls), and the chance of at least one.  The
#: design test recomputes both from the replicates, the shrunk SE ratio and the pilot noise of
#: each cell (``expected_reds``).
EXPECTED_RED_COUNT = 1.26
PROBABILITY_OF_A_RED = 0.72
GATED_CELLS = 114

#: The simulated histories per unit behind each estimate: the estimator's default, declared.
N_DRAWS = 10
#: The Monte Carlo accounting cell: one history per unit, and the independent stream of
#: :data:`M_PRIME` histories whose curve replaces the simulation term.
N_DRAWS_ONE = 1
M_PRIME = 500

LAWS = ("G1", "G2", "G3", "G4", "G5")
#: The plan labels of the shared-target and paired ICE fits, which read static plans.
STATIC_PLANS = ("treat_all", "treat_none")

#: The declared support of ``G2``'s outcome for ``LTMLE``, which scales a continuous outcome.
#: The design test shows 10^6 draws lie inside it.
G2_Q_BOUNDS = (-40.0, 50.0)

# ------------------------------------------------------------------------------- regimens


def _rule(node: int) -> Any:
    column = f"L{node}"

    def rule(history: Any) -> Any:
        return np.asarray(history[column], dtype=float).astype(int)

    return rule


def _policy(node: int) -> Stochastic:
    column = f"B{node}"
    low, slope = laws.G2_POLICY

    def probabilities(history: Any) -> pd.DataFrame:
        p = low + slope * np.asarray(history[column], dtype=float)
        return pd.DataFrame({0: 1.0 - p, 1: p})

    return Stochastic(probabilities, f"policy_b{node}", density_kind="known")


def _rule_dose(node: int) -> Any:
    column = f"L{node}"
    intercept, slope = laws.G4_RULE_DOSE

    def rule(history: Any) -> Any:
        return intercept + slope * np.asarray(history[column], dtype=float)

    return rule


def regimens(law: str, *, static_only: bool = False, with_rule: bool = False) -> dict[str, Any]:
    """The plans of one law.

    ``static_only`` keeps the static plans an ICE fit reads, and ``with_rule`` adds the dynamic
    rule of ``G1`` for the ``LTMLE`` fit that reads it.
    """
    if law == "G1":
        plans: dict[str, Any] = {
            "treat_all": 1,
            "treat_none": 0,
            "rule": DynamicRegimen("rule", tuple(_rule(t) for t in (1, 2, 3)), rule_kind="known"),
            "odds x2": Incremental(2.0, name="odds x2"),
        }
    elif law == "G2":
        plans = {
            "treat_all": 1,
            "treat_none": 0,
            "policy_b": DynamicRegimen("policy_b", tuple(_policy(t) for t in (1, 2, 3))),
        }
    elif law == "G3":
        plans = {"treat_all_4": 1, "treat_none_4": 0}
    elif law == "G5":
        plans = {"treat_all": 1, "treat_none": 0, "odds x2": Incremental(2.0, name="odds x2")}
    elif law in ("G4", "G4Q"):
        plans = {
            "natural course": Shift(0.0, cap=None),
            "shift_half": Shift(0.5, cap=None),
            "dose_half": 0.5,
            "rule_dose": DynamicRegimen(
                "rule_dose", tuple(_rule_dose(t) for t in (1, 2, 3)), rule_kind="known"
            ),
        }
        if law == "G4Q":
            # The squared dose term has a closed form only under a fixed dose.
            plans = {"dose_half": plans["dose_half"]}
    else:
        raise KeyError(law)
    if static_only:
        keep = (*STATIC_PLANS, "rule") if with_rule else STATIC_PLANS
        return {key: plans[key] for key in keep}
    return plans


REFERENCES = {
    "G1": "treat_none",
    "G2": "treat_none",
    "G3": "treat_none_4",
    "G4": "natural course",
    "G4Q": None,
    "G5": "treat_none",
}
COVARIATE_TYPES = {
    "G1": None,
    "G2": {"Z2": "normal", "Z3": "normal"},
    "G3": None,
    "G4": {"L2": "normal", "L3": "normal"},
    "G4Q": {"L2": "normal", "L3": "normal"},
    "G5": {"S2": "categorical", "S3": "categorical"},
}
HORIZONS = {"G3": [2, 4]}


def law_of(name: str) -> laws.Law:
    """The law object of a law name."""
    return {
        "G1": laws.G1,
        "G2": laws.G2,
        "G3": laws.G3,
        "G4": laws.G4,
        "G4Q": laws.G4Q,
        "G5": laws.G5,
    }[name]


# ---------------------------------------------------------------------------------- fits

#: The status of a fit whose every regression is a ``GLM`` the stacked curve reads.
INFERENTIAL = "influence_curve"


class StatusMismatch(AssertionError):
    """A fit stamped an inference status that its cell does not read.

    An ``AssertionError`` is not a recordable fitting failure
    (:func:`cleverly.validation._failures.recordable`), so it stops the run rather than
    calibrating the wrong curve.
    """


def require_status(result: Any, names: Any) -> None:
    """Raise :class:`StatusMismatch` unless the fit and each read parameter are inferential."""
    if result.inference_status != INFERENTIAL:
        raise StatusMismatch(f"fit status {result.inference_status!r}, expected {INFERENTIAL!r}")
    for name in names:
        estimate = result[name]
        if estimate.inference != INFERENTIAL or not bool(estimate.supplies_inference):
            raise StatusMismatch(
                f"{name}: status {estimate.inference!r} (supplies inference "
                f"{estimate.supplies_inference}), expected {INFERENTIAL!r}"
            )


def simulation_seed(seed: int) -> int:
    """The estimator's ``random_state``: a stream seed reduced to a valid 31-bit integer.

    The primary phase passes a seed of its ``primary_simulation`` stream, keyed by scenario and
    replicate.  The property fit sets pass a seed of their ``property_simulation`` stream.
    """
    return int(seed) % 2**31


def _columns(law: str) -> dict[str, Any]:
    return law_of(law).fit_columns(continuous_treatment=law in ("G4", "G4Q"))


def fit(
    method: str,
    law: str,
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    n_draws: int = N_DRAWS,
    n_derivative_draws: int | None = None,
    static_only: bool = False,
) -> Any:
    """One fit of ``method`` on a draw of ``law``.

    Parameters
    ----------
    method : {"nice", "ice", "ltmle", "ipsi"}
        The parametric g-formula, the ICE and ``LTMLE`` baselines on the static plans, and
        ``IPSI`` on the odds tilt.
    law : str
        A law name of :data:`LAWS` or ``"G4Q"``.
    frame : pandas.DataFrame
        The draw.
    seed : int
        The simulation seed of ``nice`` (:func:`simulation_seed`); the other methods draw no
        random number.
    n_draws, n_derivative_draws : int
        The simulated histories per unit behind the estimate and the derivatives.
    static_only : bool
        Fit only the static plans of the law.
    """
    columns = _columns(law)
    common: dict[str, Any] = {
        "treatment_learner": GLM(),
        "simultaneous": False,
    }
    if law_of(law).has_censoring:
        common["censoring_learner"] = GLM()
    with warnings.catch_warnings():
        # The shipped warnings (positivity, extrapolation) are diagnostics; none changes a
        # number, and the design test reads the support diagnostic directly.
        warnings.simplefilter("ignore")
        if method == "nice":
            model: Any = ParametricGFormula(
                regimens(law, static_only=static_only),
                reference=REFERENCES[law],
                n_draws=n_draws,
                n_derivative_draws=n_derivative_draws,
                covariate_types=COVARIATE_TYPES[law],
                horizons=HORIZONS.get(law),
                random_state=simulation_seed(seed),
                **common,
            )
        elif method in ("ice", "ltmle"):
            outcome = GLM(family="gaussian") if law == "G2" else GLM()
            kwargs: dict[str, Any] = {}
            if method == "ltmle" and law == "G2":
                kwargs["q_bounds"] = G2_Q_BOUNDS
            cls = SequentialGComputation if method == "ice" else LTMLE
            model = cls(
                regimens(law, static_only=True, with_rule=method == "ltmle" and law == "G1"),
                reference=REFERENCES[law],
                outcome_learner=outcome,
                random_state=0,
                **kwargs,
                **common,
            )
        elif method == "ipsi":
            model = IPSI(
                [Incremental(2.0, name="odds x2")],
                outcome_learner=GLM(),
                random_state=0,
                **common,
            )
        else:
            raise KeyError(method)
        return model.fit(frame, **columns)


# ------------------------------------------------------------------------------- truths


def _own_covariate(node: int) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
    return lambda table: table[f"L{node}"]


def _own_policy(
    node: int, low: float, slope: float
) -> Callable[[Mapping[str, np.ndarray]], np.ndarray]:
    return lambda table: low + slope * table[f"B{node}"]


def _node_plans(law: str, label: str) -> dict[int, laws.NodePlan]:
    nodes = range(1, law_of(law).n_times + 1)
    if label in ("treat_all", "treat_all_4"):
        return {t: laws.Fixed(1.0) for t in nodes}
    if label in ("treat_none", "treat_none_4"):
        return {t: laws.Fixed(0.0) for t in nodes}
    if label == "rule":
        return {t: laws.Rule(_own_covariate(t)) for t in nodes}
    if label == "odds x2":
        return {t: laws.Tilt(2.0) for t in nodes}
    if label == "policy_b":
        low, slope = laws.G2_POLICY
        return {t: laws.Policy(_own_policy(t, low, slope)) for t in nodes}
    raise KeyError(label)


def _linear_plan(label: str) -> dict[int, tuple[str, Any]]:
    if label == "rule_dose":
        intercept, slope = laws.G4_RULE_DOSE
        return {t: ("rule", (intercept, slope, f"L{t}")) for t in range(1, 4)}
    kind, value = {
        "natural course": ("shift", 0.0),
        "shift_half": ("shift", 0.5),
        "dose_half": ("dose", 0.5),
    }[label]
    return laws.linear_gaussian_plan(kind, value, 3)


@cache
def truths(law: str) -> Mapping[str, float]:
    """The exact truth of every reported estimand of ``law``, by estimand name.

    The binary laws sum over every branch, with Gauss-Hermite nodes for each Gaussian variable
    (:meth:`~tests.studies.parametric_gformula_laws.Law.g_formula`).  ``G4`` reads the closed
    forms of :func:`~tests.studies.parametric_gformula_laws.linear_means`.
    """
    target = law_of(law)
    if law in ("G4", "G4Q"):
        levels = {label: laws.linear_means(target, _linear_plan(label)) for label in regimens(law)}
        out = {f"ey_regimen[{label}]": value for label, value in levels.items()}
        anchor = REFERENCES[law]
        if anchor is not None:
            for label in levels:
                if label != anchor:
                    out[f"ate_regimen[{label} vs {anchor}]"] = levels[label] - levels[anchor]
        return out
    chunks = 12 if law == "G2" else 1
    if law == "G3":
        values = {
            label: target.g_formula(_node_plans(law, label), horizons=HORIZONS[law])
            for label in regimens(law)
        }
        out = {
            f"cif_regimen[{label}, {cause} @ t={horizon}]": value
            for label, by_cause in values.items()
            for (cause, horizon), value in by_cause.items()
        }
        for (cause, horizon), value in values["treat_all_4"].items():
            out[f"ate_regimen[treat_all_4 vs treat_none_4, {cause} @ t={horizon}]"] = (
                value - values["treat_none_4"][(cause, horizon)]
            )
        return out
    levels = {
        label: target.g_formula(_node_plans(law, label), chunks=chunks) for label in regimens(law)
    }
    out = {f"ey_regimen[{label}]": value for label, value in levels.items()}
    anchor = REFERENCES[law]
    assert anchor is not None
    for label in levels:
        if label != anchor:
            out[f"ate_regimen[{label} vs {anchor}]"] = levels[label] - levels[anchor]
    return out


# ------------------------------------------------------------------------------ estimands

ESTIMANDS: dict[str, tuple[str, ...]] = {
    "G1": (
        "ey_regimen[treat_all]",
        "ey_regimen[treat_none]",
        "ey_regimen[rule]",
        "ey_regimen[odds x2]",
        "ate_regimen[treat_all vs treat_none]",
    ),
    "G2": (
        "ey_regimen[treat_all]",
        "ey_regimen[treat_none]",
        "ey_regimen[policy_b]",
        "ate_regimen[treat_all vs treat_none]",
    ),
    "G3": (
        "cif_regimen[treat_all_4, relapse @ t=2]",
        "cif_regimen[treat_all_4, relapse @ t=4]",
        "cif_regimen[treat_all_4, death @ t=4]",
        "ate_regimen[treat_all_4 vs treat_none_4, relapse @ t=4]",
    ),
    "G4": (
        "ey_regimen[shift_half]",
        "ey_regimen[dose_half]",
        "ey_regimen[rule_dose]",
        "ate_regimen[shift_half vs natural course]",
    ),
    "G5": (
        "ey_regimen[treat_all]",
        "ey_regimen[treat_none]",
        "ey_regimen[odds x2]",
        "ate_regimen[treat_all vs treat_none]",
    ),
}

SCENARIOS: dict[str, tuple[str, ...]] = {f"{law.lower()}_nice": ESTIMANDS[law] for law in LAWS}

# ------------------------------------------------------------------------------- labels

#: ``law -> key -> estimand``: the estimands each law reports.
KEYS: dict[str, dict[str, str]] = {
    "G1": {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "rule": "ey_regimen[rule]",
        "odds_x2": "ey_regimen[odds x2]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    },
    "G2": {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "policy_b": "ey_regimen[policy_b]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    },
    "G3": {
        "relapse_h2": "cif_regimen[treat_all_4, relapse @ t=2]",
        "relapse_h4": "cif_regimen[treat_all_4, relapse @ t=4]",
        "death_h4": "cif_regimen[treat_all_4, death @ t=4]",
        "contrast_h4": "ate_regimen[treat_all_4 vs treat_none_4, relapse @ t=4]",
    },
    "G4": {
        "shift": "ey_regimen[shift_half]",
        "dose": "ey_regimen[dose_half]",
        "rule_dose": "ey_regimen[rule_dose]",
        "contrast": "ate_regimen[shift_half vs natural course]",
    },
    "G5": {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "odds_x2": "ey_regimen[odds x2]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    },
}


def label_of(law: str, key: str) -> str:
    """The calibration label of a law's key."""
    return f"nice_{law.lower()}_{key}"


CALIBRATION_LABELS: tuple[str, ...] = tuple(
    label_of(law, key) for law, keys in KEYS.items() for key in keys
)
LADDER_LABELS: tuple[str, ...] = (
    "nice_g1_contrast",
    "nice_g2_contrast",
    "nice_g3_contrast_h4",
    "nice_g4_contrast",
)
#: The levels the Monte Carlo accounting reads, by law.
MC_KEYS: dict[str, tuple[str, ...]] = {
    "G2": ("treat_all", "treat_none"),
    "G4": ("shift", "dose"),
    "G5": ("treat_all", "treat_none", "odds_x2"),
}
#: The laws whose accounting fit set also fits the independent ``m'`` stream and reports the
#: ``mc_removed`` cell.  ``G5`` reports the ``n_draws_1`` cell only: its ``m'`` fit costs 2.2 s per
#: replicate for a cell that carries no verdict.
MC_REMOVED_LAWS: tuple[str, ...] = ("G2", "G4")
MC_LABELS: tuple[str, ...] = tuple(
    label_of(law, key) for law, keys in MC_KEYS.items() for key in keys
)
#: The paired ICE labels of ``G2`` that the necessity family gates.  The design test shows each
#: has predictive power at least 0.95 at :data:`PAIRED_N`.
ICE_GATED: tuple[str, ...] = ("pair_g2_treat_all", "pair_g2_treat_none", "pair_g2_contrast")
#: Every paired ICE label, by law: ``key -> estimand``.
PAIR_KEYS: dict[str, dict[str, str]] = {
    "G1": {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    },
    "G2": {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    },
}
PAIR_LABELS: tuple[str, ...] = tuple(
    f"pair_{law.lower()}_{key}" for law, keys in PAIR_KEYS.items() for key in keys
)
#: The shared-target labels: ``law -> key -> (estimand, other method)``.
SHARED_KEYS: dict[str, dict[str, tuple[str, str]]] = {
    "G1": {
        "treat_all": ("ey_regimen[treat_all]", "ltmle"),
        "contrast": ("ate_regimen[treat_all vs treat_none]", "ltmle"),
        "rule": ("ey_regimen[rule]", "ltmle"),
        "odds_x2": ("ey_regimen[odds x2]", "ipsi"),
    },
    "G2": {
        "treat_all": ("ey_regimen[treat_all]", "ltmle"),
        "contrast": ("ate_regimen[treat_all vs treat_none]", "ltmle"),
    },
}
SHARED_LABELS: tuple[str, ...] = tuple(
    f"target_{law.lower()}_{key}" for law, keys in SHARED_KEYS.items() for key in keys
)
EXTRAPOLATION_LABEL = "g4q_dose"


#: Every published property cell, by family.  :mod:`tests.studies.parametric_gformula_properties`
#: declares how each is drawn and read.
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "root_n_and_efficiency": tuple(
        f"{label}__n_{size}" for label in LADDER_LABELS for size in LADDER
    ),
    "root_n_rate": tuple(
        f"{label}__{suffix}"
        for label in LADDER_LABELS
        for suffix in ("empirical_sd", "reported_se")
    ),
    "monte_carlo_accounting": tuple(
        f"{label}__{kind}"
        for label in MC_LABELS
        for kind in (
            ("n_draws_1", "mc_removed")
            if label.split("_")[1].upper() in MC_REMOVED_LAWS
            else ("n_draws_1",)
        )
    ),
    "ice_necessity": tuple(f"{label}__{arm}" for label in ICE_GATED for arm in ("nice", "ice")),
    "ice_reported": tuple(
        f"{label}__{arm}"
        for label in PAIR_LABELS
        if label not in ICE_GATED
        for arm in ("nice", "ice")
    ),
    "shared_target": tuple(
        f"target_{law.lower()}_{key}__{arm}"
        for law, shared in SHARED_KEYS.items()
        for key, (_, method) in shared.items()
        for arm in ("nice", method)
    ),
    "extrapolation_report": (f"{EXTRAPOLATION_LABEL}__main_terms",),
}
PROPERTY_CELLS = {family: cells for family, cells in PROPERTY_CELLS.items() if cells}

STUDY = StudyRecord(
    name="parametric g-formula",
    slug="parametric-gformula",
    artifacts=ROOT / "tests" / "canonical" / "parametric_gformula_study",
    document="docs/technical-reference/method-evidence/parametric-gformula.md",
    anchor="parametric-g-formula",
    scenarios=dict(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    calibration_efficiency_ratio=False,
    modules=(
        "tests/studies/canonical_parametric_gformula.py",
        "tests/studies/parametric_gformula_properties.py",
        "tests/studies/parametric_gformula_laws.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_parametric_gformula",
    properties_module="tests.studies.parametric_gformula_properties",
    property_cells=PROPERTY_CELLS,
    nuisance_count=3,
    publication_policy="reporting",
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
)

CONFIGURATION = {
    "estimator": "ParametricGFormula",
    "covariate_learner": "GLM() main terms, one model per covariate and node",
    "outcome_learner": "GLM() of the outcome's family",
    "treatment_learner": "GLM()",
    "n_draws": N_DRAWS,
    "monte_carlo_control": {"n_draws": N_DRAWS_ONE, "m_prime": M_PRIME},
    "simulation_stream": "Philox keyed by the primary_simulation stream (scenario, replicate) in the primary "
    "phase and the property_simulation stream (fit set, replicate) in the property fit sets",
    "ladder": list(LADDER),
    "calibration_n": CALIBRATION_N,
    "reference": f"none; pygformula {PYGFORMULA} runs on the fixtures of compare_pygformula.py",
}

# --------------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``scenario``'s law, with its exact truths."""
    law = scenario.split("_", 1)[0].upper()
    return law_of(law).sample(n, seed), dict(truths(law))


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": IMPLEMENTATION,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(estimate.psi),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    law = scenario.split("_", 1)[0].upper()
    seed = stream_seed(STUDY, "primary_simulation", scenario, replicate)
    result = fit("nice", law, frame, seed=seed)
    require_status(result, SCENARIOS[scenario])
    return result_rows(result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=IMPLEMENTATION,
        rows_at=None,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    estimates = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)])
