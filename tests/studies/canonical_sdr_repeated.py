"""Canonical repeated-outcome ``SDR`` evidence study (X32b, PR B).

``SDR`` fits the repeated outcome ``Y_1, ..., Y_T`` of
:func:`cleverly.datasets.make_longitudinal_repeated` once, with
``outcome=Repeated(["Y1", "Y2", "Y3"])``, and reports the mean of ``Y_h`` under three plans at
``h = 1, 2, 3``.  Each horizon is the end-of-study fit of the panel truncated after node ``h``
(Díaz, Williams, Hoffman and Schenck 2023, Equation (4), Lemmas 3 and 4 and Theorem 4 on the
prefix).  The truth-calibrated property study lives in
:mod:`tests.studies.sdr_repeated_properties`.

The comparator is pending
-------------------------

R ``lmtp`` 1.5.4 (``cf_sdr``) pairs with this study on the same draws.  The lmtp runs are not part
of this change.  The record declares ``pending_reference="lmtp"``, so the regeneration driver runs
no reference process, the registry holds the study as "comparator pending", and the evidence is
labelled "Python property evidence, comparator run pending".  The merge reads the Python rows
only: the truth gates on the 26 cleverly cells, and the fast-tier witnesses E2, E3 and N8' that
carry the correction's exactness without R.  This change writes the pairing columns so that the
change which registers lmtp can pair on the draws committed here (plan X32b, sections 5.3 and 5.5):

* ``ratio__<plan>__<node>``: the per-node ratio :math:`\\pi_t 1\\{C_t = 1\\} / (g_t c_t)` at the
  observed arm, zero where the unit leaves the plan.  :func:`check_ratios` asserts that its
  running products are the partial weights the engine reads.  The mechanisms are known
  (:class:`~tests.studies.canonical_ltmle_repeated.KnownRepeatedMechanism`), so the ratios do not
  depend on the fold, and the training-fold recursion equals ``lmtp::cf_sdr``;
* ``shift__<plan>__<node>__1``: the arm each unit takes at the node under the plan;
* ``fold``: the fold assignment of the fit, which the comparator receives as ``task$folds``.

lmtp has no non-absorbing repeated-outcome mode.  Its pairing runs once per ``(plan, horizon)`` on
the frame truncated after node ``h``.  That is the X32a reduction, applied by the study.

Declared before any run
-----------------------

* Scenarios ``repeated_folds_1`` (in sample) and ``repeated_folds_5``.  Both draw independent
  samples, so there is no seed owner and the bias clause reads one group per scenario.
* Plans ``never``, ``always`` and the X32a rule (treat at node 1, then treat after
  :math:`Y_{t-1} = 1`).  Horizons 1 to 3.  Thirteen estimands: eight means (the rule equals
  ``always`` at ``t = 1``) and five contrasts against ``never``.  Twenty-six scenario cells.
* The outcome regression and every earlier-node regression is the pairing learner
  :class:`CellMeansOnOutcomes`: cell means keyed on :math:`Y_{<t}` among the plan's followers
  through ``t``.  The key omits ``W`` and ``L``, so the regression is wrong, the correction is
  nonzero in every replicate at both fold counts, and the pairing tests something.  Saturated
  cell means would make the in-sample correction vanish.  The key must hold :math:`Y_{<t}`
  because a rule follower is defined by :math:`Y_{<t}` and :math:`A_{\\le t}` together.
* The mechanisms are the generating ones (``KnownRepeatedMechanism``) with
  ``g_bounds = G_BOUNDS``.  With known ratios ``SDR`` is consistent and its curve is exact to
  first order (:math:`E[\\varphi_1(\\bar m, r)] = \\psi` for any :math:`\\bar m`), so the truth and
  coverage gates are valid.  The population limit is the truth in every plan for any key.
* Truth-gate necessity (plan L3).  The pairing learner's plug-in at ``n = 12,000`` misses the
  truth by more than 0.25 EIF-SE in 11 of 13 cells.  The rule's two means miss by 0.15 and 0.24,
  too small to count.  A fit that skipped the correction would fail the bias gate in the 11 cells.
  A fast test pins all 13 gaps.
* ``PRIMARY_N = 12,000`` is the smallest multiple of 1,000 at which the expected number of
  unseen-cell fits over the run is at most 0.01 (:func:`expected_unseen_cell_fits`).  An unseen
  cell makes lmtp return ``NA``.  A fast test recomputes it at the final ``PRIMARY_REPLICATES``.
  The ``n`` rule of :mod:`tests.studies.repeated_sizing` (``n_rule_fires``) may double it, twice
  at most, and the unseen-cell count then still holds.
* ``PRIMARY_REPLICATES`` is :func:`tests.studies.sdr_repeated_properties.primary_replicates`
  applied to pilot P1: the smallest multiple of 400 that meets clauses (a) to (d) of
  :func:`tests.studies.repeated_sizing.primary_replicates` at the share
  :data:`~tests.studies.repeated_sizing.PRIMARY_CLAUSE_SHARE`, with two implementations (the
  lmtp rows of the later change fit the same ``R``).
* The cleverly rows publish the first-node plug-in as ``initial_estimate``: the mean of the first
  node's regression of the corrected chain.  It is the quantity the lmtp adapter returns beside
  its estimate, so the parity ratio of the later change can read the correction on both sides.
* The registered :class:`Margins` defaults apply.  Publication policy ``gated``.  A red Python
  cell is diagnosed and fixed before a fresh declaration on fresh seeds.  No red cell is deferred.

Pilot state
-----------

:data:`PILOT_STATE` is ``"placeholder"`` until pilots P1, P2 and P3 are run
(``tests/canonical/sdr_repeated/README.md`` gives the commands).  While it is a placeholder, the
pilot tables in :mod:`tests.studies.sdr_repeated_properties` hold declared uniform planning
values, the sizes in this module are the rules applied to them, and a declared run is refused
(:func:`require_recorded_pilots`).  The disposable smoke run with explicit ``--replicates`` and
``--n`` is not a declared run.  The study stays unregistered until its artifacts and its row of
the implementation validation grid exist.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import SDR
from cleverly.longitudinal.regimen import (
    Plan,
    attach_policy_numerators,
    resolve_plans,
    resolve_regimens,
)
from cleverly.longitudinal.sdr import node_factors, partial_weights
from tests.discrete_law_longitudinal import CellMeans
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.canonical_ltmle import G_BOUNDS, regimen_rows
from tests.studies.canonical_ltmle_repeated import (
    COLUMNS,
    EARLIER_LABELS,
    ESTIMANDS,
    GATED_LABELS,
    HORIZONS,
    REGIMENS,
    TRUTH,
    KnownRepeatedMechanism,
    label_estimand,
)
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.ltmle_repeated_properties import (
    ABSORBING_LABELS,
    HISTORY_LABELS,
    RULE_LABELS,
    CellMeansOn,
)

#: ``"placeholder"`` while the sizes below are the rules applied to declared uniform planning
#: values, and ``"recorded"`` once pilots P1, P2 and P3 are pinned in the properties module.
PILOT_STATE: Literal["placeholder", "recorded"] = "placeholder"

#: The smallest multiple of 1,000 whose expected unseen-cell count is at most
#: :data:`UNSEEN_CELL_BUDGET` at the declared replicates.  A fast test re-derives it.
PRIMARY_N = 12_000
#: From ``sdr_repeated_properties.primary_replicates`` on the primary pilot (see the module
#: docstring, Pilot state).  A fast test re-derives it.
PRIMARY_REPLICATES = 2_400
SEED = 20263401
RESAMPLING_SEED = 2026340101
SCENARIOS = ("repeated_folds_1", "repeated_folds_5")
#: The folds of each scenario's fit.
SCENARIO_FOLDS = {"repeated_folds_1": 1, "repeated_folds_5": 5}
#: The most unseen-cell fits the run may expect.  lmtp returns ``NA`` on an unseen cell.
UNSEEN_CELL_BUDGET = 0.01
REFERENCE = "never"

#: The primary labels, in the order of :data:`ESTIMANDS`: the thirteen unique parameters.  The
#: rule's ``t = 1`` mean and contrast duplicate ``always``'s and are not primary cells.
_ESTIMAND_LABEL = {label_estimand(name): name for name in (*GATED_LABELS, *EARLIER_LABELS)}
LABELS = tuple(_ESTIMAND_LABEL[name] for name in ESTIMANDS)

# ---------------------------------------------------------------------------- the property cells

#: The property families and their arms.  Each arm is a suffix of ``label__arm``.
CALIBRATION_ARMS = ("in_sample", "crossfit", "continuous")
BOOSTER_ARM = "crossfit_boosted"
SMALL_SAMPLE_ARMS = ("in_sample", "continuous")
DOUBLE_ROBUSTNESS_ARMS = ("mechanism_correct", "outcome_correct", "both_wrong")
CORRECTION_ARMS = ("corrected", "uncorrected")
RULE_ARMS = ("declared", "reversed_rule")
ABSORBING_ARMS = ("repeated", "absorbing_reading")
HISTORY_FREE_ARMS = ("repeated", "history_free")
FULL_HISTORY_ARMS = ("full_history", "markov_history")
JOINT_LABEL = "all_reported"


def _property_cells() -> dict[str, tuple[str, ...]]:
    """The property cells ``sdr_repeated_properties`` publishes, by family."""
    return {
        "interval_calibration": (
            *(f"{name}__in_sample" for name in LABELS),
            *(f"{name}__crossfit" for name in GATED_LABELS),
            *(f"{name}__continuous" for name in GATED_LABELS),
            *(f"{name}__{BOOSTER_ARM}" for name in GATED_LABELS),
            *(
                f"{name}__{control}"
                for name in LABELS
                for control in ("shrunken_se_control", "noise_control")
            ),
        ),
        "small_sample_calibration": tuple(
            f"{name}__{arm}" for name in GATED_LABELS for arm in SMALL_SAMPLE_ARMS
        ),
        "double_robustness": tuple(
            f"{name}__{arm}" for name in GATED_LABELS for arm in DOUBLE_ROBUSTNESS_ARMS
        ),
        "correction_necessity": tuple(
            f"{name}__{arm}" for name in GATED_LABELS for arm in CORRECTION_ARMS
        ),
        "simultaneous_coverage": tuple(
            f"{JOINT_LABEL}__{kind}" for kind in ("simultaneous_band", "pointwise_joint_control")
        ),
        "rule_necessity": tuple(f"{name}__{arm}" for name in RULE_LABELS for arm in RULE_ARMS),
        "repeated_outcome_necessity": tuple(
            f"{name}__{arm}" for name in ABSORBING_LABELS for arm in ABSORBING_ARMS
        ),
        "outcome_history_necessity": tuple(
            f"{name}__{arm}" for name in HISTORY_LABELS for arm in HISTORY_FREE_ARMS
        ),
        "full_history_necessity": tuple(
            f"{name}__{arm}" for name in GATED_LABELS for arm in FULL_HISTORY_ARMS
        ),
    }


# ---------------------------------------------------------------------------- the record

STUDY = StudyRecord(
    name="repeated-outcome longitudinal SDR",
    slug="longitudinal-repeated-sdr",
    outcome_kind="repeated",
    artifacts=ROOT / "tests" / "canonical" / "sdr_repeated",
    document="docs/technical-reference/method-evidence/repeated-outcome-sdr.md",
    anchor="repeated-outcome-longitudinal-sdr",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly",
    # The R lmtp 1.5.4 pairing runs on the owner's machine as the TRACKER Study queue item that
    # the change registering the lmtp rows (X32b PR C, TRACKER row 12c) adds.  It does not block
    # the merge, and that change moves this name to ``reference``.
    pending_reference="lmtp",
    pending_reason=(
        "the R lmtp 1.5.4 paired runs (cf_sdr, one call per plan and horizon on the truncated "
        "prefix, on the draws and pairing columns committed here) run on the owner's machine as "
        "the TRACKER Study queue item of X32b PR C; that change registers them"
    ),
    modules=(
        "tests/studies/canonical_sdr_repeated.py",
        "tests/studies/sdr_repeated_properties.py",
        "tests/studies/repeated_sizing.py",
        # The law, the learners and the helpers the two modules import.
        "tests/studies/ltmle_repeated_law.py",
        "tests/studies/ltmle_repeated_properties.py",
        "tests/studies/canonical_ltmle_repeated.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/fractional_glm.py",
        "src/cleverly/datasets/longitudinal.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.canonical_sdr_repeated",
    properties_module="tests.studies.sdr_repeated_properties",
    property_cells=_property_cells(),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_sdr and theta_dr(is_sdr = TRUE), one call per plan and horizon on the panel "
        "truncated after node h (binomial outcome Y_h, the earlier Y_s as covariates), with the "
        "per-node ratios, shifted values and fold assignment written beside each replicate; "
        "the outcome learner is cell means keyed on (Y_<t, A_<=t); no lmtp density ratio is "
        "fitted"
    ),
}


class PlaceholderPilotsError(RuntimeError):
    """A declared run was requested while the pilot tables are placeholders."""


def require_recorded_pilots(what: str) -> None:
    """Refuse ``what`` while :data:`PILOT_STATE` is ``"placeholder"``.

    A declared run draws the sizes this module and the properties module state.  While the pilot
    tables are declared uniform planning values, those sizes are not the rules applied to a
    pilot, and a run under them would publish verdicts that no pilot sized.
    """
    if PILOT_STATE != "recorded":
        raise PlaceholderPilotsError(
            f"{what} is refused: pilots P1, P2 and P3 have not been run, so "
            f"{STUDY.slug} holds placeholder pilot tables "
            "(tests/canonical/sdr_repeated/README.md gives the pilot commands)"
        )


# ---------------------------------------------------------------------------- the pairing learner


class CellMeansOnOutcomes(CellMeansOn):
    r"""``learner`` fitted on the earlier outcomes :math:`Y_{<t}` of the outcome design alone.

    The pairing learner of the primary.  A repeated-outcome fit hands the regression at node
    ``t`` the design :math:`[W, Y_1, L_2, \ldots, Y_{t-1}, L_t]`, whose width is 1, 3 or 5.  The
    key keeps the :math:`Y` columns (positions 1, 3, ...) and drops ``W`` and ``L``.  At node 1
    the key is empty, and the regression is the mean among the followers.  The plan's
    followers carry no arm column, so the key :math:`(Y_{<t}, A_{\le t})` of the lmtp side reads
    the same cell at the prediction point.

    Parameters
    ----------
    learner : estimator
        The learner fitted on the selected columns.
    """

    def __init__(self, learner: Any) -> None:
        super().__init__("outcomes", learner)

    @staticmethod
    def columns(kind: str, width: int) -> list[int]:
        """The design columns kept at ``width``: the earlier outcomes for ``"outcomes"``."""
        if kind == "outcomes":
            return list(range(1, width, 2))
        return CellMeansOn.columns(kind, width)

    def _select(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        kept = self.columns(self.kind, matrix.shape[1])
        # A constant column carries node 1, whose key is empty.
        return np.column_stack([np.zeros(len(matrix)), matrix[:, kept]])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"learner": self.learner}

    def set_params(self, **params: Any) -> CellMeansOnOutcomes:
        if "learner" in params:
            self.learner = params.pop("learner")
        if params:
            raise ValueError(f"invalid parameters: {sorted(params)}")
        return self


# ---------------------------------------------------------------------------- the unseen-cell rule


def cell_masses() -> np.ndarray:
    """The mass of every :math:`Y_{<t}` cell among the followers of a plan through ``t``.

    Plans ``never``, ``always`` and the rule, nodes ``t = 2`` and ``t = 3``, on the binary law.
    Node 1 has an empty key, so it holds no cell that a fit can miss.
    """
    paths = law.BINARY.paths
    weights = paths["p"].to_numpy()
    masses: list[float] = []
    for plan in ("never", "always", law.RULE):
        for time in (2, 3):
            followed = law.followed_mask(paths, plan, time)
            grouped = (
                paths[followed]
                .assign(p=weights[followed])
                .groupby([f"Y{s}" for s in range(1, time)])["p"]
                .sum()
            )
            masses.extend(grouped.to_numpy())
    return np.array(masses)


def expected_unseen_cell_fits(replicates: int, n: int) -> float:
    """The expected number of fits over the run in which a cell has no follower.

    The one-fold scenario fits all ``n`` rows, so a cell of mass :math:`m` is unseen with
    probability :math:`(1 - m)^n`.  The five-fold scenario fits five training folds of
    :math:`0.8 n` rows.  Both scenarios draw ``replicates`` fits (plan X32b, section 5.3).
    """
    masses = cell_masses()
    one_fold = float(np.sum((1.0 - masses) ** n))
    five_fold = 5.0 * float(np.sum((1.0 - masses) ** (0.8 * n)))
    return replicates * (one_fold + five_fold)


def minimum_primary_n(replicates: int) -> int:
    """The smallest multiple of 1,000 with at most :data:`UNSEEN_CELL_BUDGET` expected misses."""
    n = 1_000
    while expected_unseen_cell_fits(replicates, n) > UNSEEN_CELL_BUDGET:
        n += 1_000
    return n


# ---------------------------------------------------------------------------- the fit


def fit_cleverly(frame: pd.DataFrame, *, folds: int, simultaneous: bool = False) -> Any:
    """The primary fit: ``SDR`` with the pairing learner and the known mechanisms."""
    return SDR(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=CellMeansOnOutcomes(CellMeans()),
        pseudo_learner=CellMeansOnOutcomes(CellMeans()),
        treatment_learner=KnownRepeatedMechanism("treatment"),
        censoring_learner=KnownRepeatedMechanism("censoring"),
        n_folds=folds,
        g_bounds=G_BOUNDS,
        simultaneous=simultaneous,
        random_state=0,
    ).fit(frame, **COLUMNS)


#: The primary fit, recorded in the manifest.
CONFIGURATION = {
    "construction": "sdr",
    "outcome_kind": "repeated",
    "horizon_mode": "all_prefixes",
    "estimator": "sequentially_doubly_robust",
    "scenario_folds": dict(SCENARIO_FOLDS),
    "outcome_learner": "CellMeansOnOutcomes(CellMeans): cell means keyed on Y_<t",
    "pseudo_learner": "CellMeansOnOutcomes(CellMeans): cell means keyed on Y_<t",
    "mechanism": "known, KnownRepeatedMechanism",
    "reference": REFERENCE,
    "g_bounds": list(G_BOUNDS),
    "horizons": list(HORIZONS),
    "regimens": list(REGIMENS),
    "labels": list(LABELS),
    "pairing_columns": ["fold", "ratio__<plan>__<node>", "shift__<plan>__<node>__1"],
}


# ---------------------------------------------------------------------------- pairing columns


def slug(label: str) -> str:
    """The column-name form of a plan label."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def plans_of(result: Any) -> tuple[Plan, ...]:
    """The fit's plans, resolved again with the mechanism's policy numerators attached."""
    data = result.data
    specs = resolve_regimens(
        dict(REGIMENS), data.n_decisions, held_nodes=data.n_times if data.treatment_held else None
    )
    return attach_policy_numerators(resolve_plans(specs, data), result.mechanism.policy_numerators)


def supplied_ratios(result: Any, plan: Plan) -> np.ndarray:
    r"""``(n, T)`` per-node ratio :math:`\pi_t 1\{C_t = 1\} / (g_t c_t)` at the observed arm.

    Zero where the unit leaves the plan, or the observation, at node :math:`t`.  The mechanism is
    the fit's own, which is fold-free because it is known.
    """
    data = result.data
    numerator = plan.intervention_density(data)
    factors = node_factors(data, plan, result.mechanism)
    observed = np.asarray(data.uncensored, dtype=bool) & (numerator > 0.0)
    safe = np.where(factors > 0.0, factors, 1.0)
    return np.where(observed & (factors > 0.0), numerator / safe, 0.0)


def check_ratios(result: Any, plan: Plan, ratios: np.ndarray) -> None:
    """The running products of ``ratios`` are the unbounded weights :math:`W_{1,s}`."""
    data = result.data
    numerator = plan.intervention_density(data)
    factors = node_factors(data, plan, result.mechanism)
    masks = plan.masks(data)
    weights, _, binding = partial_weights(
        numerator, factors, masks, 1, data.n_times, (np.finfo(float).tiny, 1.0)
    )
    if binding.any():
        raise AssertionError(f"plan {plan.label!r}: a partial path probability is zero")
    running = np.ones(data.n)
    for node, weight in enumerate(weights, start=1):
        running = running * ratios[:, node - 1]
        following = masks.following(node)
        if not np.allclose(running[following], weight[following], rtol=1e-12, atol=0.0):
            raise AssertionError(
                f"plan {plan.label!r} at node {node}: the supplied ratios are not the weights"
            )


def shifted_values(result: Any, plan: Plan, node: int) -> np.ndarray:
    """The arm each unit takes at ``node`` under ``plan``, for the comparator's shifted values."""
    if plan.policy_targets or plan.mtp_assignments or plan.policy:
        raise AssertionError(f"plan {plan.label!r}: only a label or a rule is paired")
    levels = np.asarray(result.data.treatment_levels[node - 1], dtype=object)
    codes = np.nan_to_num(np.asarray(plan.values[:, node - 1], dtype=float)).astype(int)
    return levels[codes]


def pairing_columns(result: Any) -> dict[str, np.ndarray]:
    """The ``ratio__`` and ``shift__`` columns of every plan, for the comparator to read."""
    out: dict[str, np.ndarray] = {}
    for plan in plans_of(result):
        ratios = supplied_ratios(result, plan)
        check_ratios(result, plan, ratios)
        name = slug(plan.label)
        for node in range(1, result.data.n_times + 1):
            out[f"ratio__{name}__{node}"] = ratios[:, node - 1]
            out[f"shift__{name}__{node}__1"] = shifted_values(result, plan, node)
    return out


# ---------------------------------------------------------------------------- rows


def _first_node_mean(result: Any, plan: str, horizon: str) -> float:
    """The mean of the first node's regression of ``plan``'s fit at ``horizon``."""
    return float(np.mean(result.fits[f"{plan} @ t={horizon}"].steps[0].initial))


def initials(result: Any) -> dict[str, float]:
    """The first-node plug-in of each estimand: the mean of step one's regression.

    A contrast is the difference of its two means.
    """
    out: dict[str, float] = {}
    for name in ESTIMANDS:
        if name.startswith("ey_regimen["):
            plan, horizon = name[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
            out[name] = _first_node_mean(result, plan, horizon)
        else:
            comparison, horizon = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
            left, right = comparison.split(" vs ")
            out[name] = _first_node_mean(result, left, horizon) - _first_node_mean(
                result, right, horizon
            )
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in SCENARIO_FOLDS:
        raise KeyError(scenario)
    frame, _ = make_longitudinal_repeated(n=n, seed=seed, censoring=True, backend="pandas")
    return frame, dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    n: int,
) -> list[dict[str, Any]]:
    """One replication's rows of a fitted ``result``."""
    if scenario not in SCENARIO_FOLDS:
        raise KeyError(scenario)
    return regimen_rows(STUDY, result, truth, initials(result), ESTIMANDS, scenario, replicate, n=n)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit_cleverly(frame, folds=SCENARIO_FOLDS[scenario])
    sample = frame.copy()
    for name, values in pairing_columns(result).items():
        sample[name] = values
    sample.insert(0, "row", np.arange(len(sample)))
    sample.insert(0, "fold", result.folds.assignment)
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truths = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in truth.items()
    ]
    return sample, truths, cleverly_rows(result, truth, scenario, replicate, n=len(frame))


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication, and write the pairing columns.

    The declared ``(replicates, n)`` is refused while the pilot tables are placeholders.  An
    explicit smaller run is a smoke run and is not refused.  The sample of each replicate carries
    the fold assignment and the pairing columns of the fit, so a failed fit leaves its replicate
    without a sample.
    """
    if (replicates, n) == (PRIMARY_REPLICATES, PRIMARY_N):
        require_recorded_pilots("the declared primary run")
    # Replicate-major, so every scenario of one replicate is contiguous in the sample archive and
    # the comparator's runner reads them as one group.
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, n)
            for replicate in range(replicates)
            for scenario in STUDY.scenarios
        ],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truths = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], samples=samples, truths=truths
    )
