r"""Property families of ``survival-node-bounds`` (X33 part (i), ``X22-plan.md`` II-3.4).

Every fit is :func:`tests.studies.canonical_survival_node_bounds.fit`: in sample, with the
declared learners, ``hazard_pooling="pooled"`` and the valid bounds of
:data:`~tests.studies.canonical_survival_node_bounds.BOUND_FACTOR`, unless the table says
otherwise.  One draw of the calibration stream is fitted four times: the bounded and the
unbounded fit of each model.  So K, W and C3 read the same draws.

==============================  ===============================================================
family                          cells, replications and size
==============================  ===============================================================
``interval_calibration`` (K)    ``F(5 | 1)`` and ``RD(5)`` of the bounded mean fit and of the
                                bounded hazard fit (four positives), each with its derived
                                ``shrunken_se_control`` and ``noise_control`` and the exact-EIF
                                efficiency band; n = 2,000, :data:`CALIBRATION_REPLICATES`
``reported_calibration``        ``F(3 | 1)``, ``F(3 | 0)`` and ``F(5 | 0)`` of both bounded
                                fits on the same draws (diagnostic, no verdict)
``binding_witness`` (W)         per model: the share of draws on which the unbounded fit leaves
                                the declared bounds on some node-``t`` follower, and the share
                                on which the bounded and unbounded estimates of ``F(5 | 1)`` or
                                ``RD(5)`` differ by more than :data:`WITNESS_MOVEMENT`; each
                                share at least
                                :data:`WITNESS_SHARE`.  The mean model gates; the hazard model
                                reports (:data:`WITNESS_GATED`)
``unbounded_fit`` (C3)          ``F(5 | 1)`` and ``RD(5)`` of each unbounded fit on the same
                                draws, with the RMSE of the bounded and unbounded fits side by
                                side (diagnostic; Benkeser, Carone and Gilbert 2018, Section 5)
``bound_necessity`` (N)         ``F(5 | 1)`` of the bounded mean fit with an intercept-only
                                treatment mechanism and a saturated outcome regression
                                (``trt*W1*W2``), at :data:`NECESSITY_N`: the valid bounds
                                (positive) and :data:`NECESSITY_BOUNDS`, which exclude the truth
                                in the high-risk cells (control)
==============================  ===============================================================

**Rules.**  K reads the shared calibration rule with :data:`EFFICIENCY_RATIO_BAND` around the
exact efficient influence function (:data:`EFFICIENCY_SD`).  N reads
:func:`~tests.studies.evidence.property_verdicts.necessity_verdicts`: the positive arm's bias
is equivalent to zero, the control's bias is discriminated beyond the margin, and their paired
displacement is at least :data:`NECESSITY_DISPLACEMENT` of the positive arm's SD.  W reads its
two shares against :data:`WITNESS_SHARE` (S10, a share filled at its adverse support bound).
Every gated cell also reads the failure rule of the declaration: more than
:data:`UNCONVERGED_SHARE` unconverged replicates turn the cell red (:func:`convergence_verdicts`).
:data:`PROPERTY_RULES` declares it as one more clause of each gated family's rule, so the
failure accounting and the property differential restate it.  The diagnostic families publish
their numbers and state no verdict.

**N's control** sets the upper bound at every node to :data:`NECESSITY_SHARE` of the way from
the arm-1 at-risk follower mean of the true iterated mean to its maximum over ``W``
(:func:`excluding_bounds`).  A root then exists at every node, so the fit converges, while
the bound excludes the truth in the high-risk cells.  The declaration names node 5 alone.  At
node 5 alone the exact-law limit is about 0.1 SD at n = 2,000, below the margin 0.25, so the
design test widened the control to every node.  The sampled follower mean must also stay
below the lowered bound, which needs a gap of several of its standard errors.  That sets
:data:`NECESSITY_N` and :data:`NECESSITY_SHARE`: at n = 16,000 and share 0.5 the exact-law
limit is 0.59 SD, and the smallest root gap is 3.8 of its standard deviations
(``tests/unit/test_survival_node_bounds_design.py``; the design note records the deviation).
N reads the study's one replication count.

A replication that raises is not redrawn; the failure accounting of
``docs/development/method-benchmarking.md`` applies at :data:`FAILURE_BOUND`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from tests.parallel import STUDY_JOBS
from tests.studies import canonical_survival_node_bounds as study
from tests.studies import survival_node_bounds_law as law
from tests.studies.evidence import property_failures, property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.replication import fit_properties
from tests.studies.evidence.seeds import stream_seed
from tests.studies.point_survival_common import CellMeans
from tests.studies.point_survival_properties import MarginalProbabilities

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CALIBRATION_N = study.PRIMARY_N
CALIBRATION_REPLICATES = study.REPLICATES
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The W shares and the movement floor (``X22-plan-review.md`` finding 5).
WITNESS_SHARE = 0.5
WITNESS_MOVEMENT = 1e-4
#: The models whose W cell gates; the others report their shares.
WITNESS_GATED = (study.MEAN_MODEL,)

#: N: the size, the replications, the control's share of the gap, and the displacement floor.
NECESSITY_N = 16_000
NECESSITY_REPLICATES = CALIBRATION_REPLICATES
NECESSITY_SHARE = 0.5
NECESSITY_NODES = (1, 2, 3, 4, 5)
NECESSITY_DISPLACEMENT = 0.25

F5_ARM1 = law.risk_name("arm1", 5)
F5_ARM0 = law.risk_name("arm0", 5)
F3_ARM1 = law.risk_name("arm1", 3)
F3_ARM0 = law.risk_name("arm0", 3)
RD5 = law.contrast_name(5)

#: ``label -> (model, estimand)`` of the gated calibration cells.
CALIBRATION: dict[str, tuple[str, str]] = {
    f"{model}_{key}": (model, name)
    for model in study.MODELS
    for key, name in (("f5_arm1", F5_ARM1), ("rd5", RD5))
}
CALIBRATION_LABELS = tuple(CALIBRATION)
#: ``label -> (model, estimand)`` of the reported calibration cells.
REPORTED: dict[str, tuple[str, str]] = {
    f"{model}_{key}": (model, name)
    for model in study.MODELS
    for key, name in (("f3_arm1", F3_ARM1), ("f3_arm0", F3_ARM0), ("f5_arm0", F5_ARM0))
}
#: ``label -> (model, estimand)`` of the unbounded cells (C3).
UNBOUNDED = CALIBRATION
EFFICIENCY_SD: dict[str, float] = {
    label: law.EFFICIENCY_SD[name] for label, (_, name) in {**CALIBRATION, **REPORTED}.items()
}

#: The calibration positive arm of each label to the controls derived from it.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "reported_calibration": tuple(f"{label}__reported" for label in REPORTED),
    "binding_witness": tuple(f"{model}__binding" for model in study.MODELS),
    "unbounded_fit": tuple(f"{label}__unbounded" for label in UNBOUNDED),
    "bound_necessity": ("f5_arm1__valid_bounds", "f5_arm1__excluding_bound"),
}

#: The extra row columns of W.
WITNESS_COLUMNS = ("exceeds_bound", "estimate_moved")
#: The per-row convergence flag, and the largest unconverged share a cell may carry
#: (``X22-plan.md`` II-3.4, failure rule).
CONVERGENCE_COLUMN = "converged"
UNCONVERGED_SHARE = 0.01
#: The gated families the convergence rule reads, each with the group of its joint verdict.
#: The rule is one more clause of each family's declared rule (:data:`PROPERTY_RULES`), so a
#: cell over the share fails its own verdict and every joint verdict its family shares.  The
#: diagnostic families and the diagnostic hazard witness state no verdict, so the rule does not
#: read them.  They still publish their share.
CONVERGENCE_GROUPS: dict[str, str] = {
    "interval_calibration": "row",
    "bound_necessity": "family",
    "binding_witness": "family",
}


# ---------------------------------------------------------------------------- N


def at_risk_mean(t: int, a: int, horizon: int = 5) -> float:
    """The follower mean of the true iterated mean at node ``t`` in arm ``a``.

    The followers at node ``t`` are the units of arm ``a`` with no event before ``t`` and
    still observed.  The censoring weights of the clever covariate undo the censoring, and the
    intercept-only treatment mechanism is constant within an arm, so the H-weighted follower
    mean is this at-risk mean.
    """
    num = den = 0.0
    for cell in law.CELLS:
        alive = 1.0 - law.iterated_mean(1, t - 1, a, *cell) if t > 1 else 1.0
        weight = law.p_w(*cell) * law.g(a, *cell) * alive
        num += weight * law.iterated_mean(t, horizon, a, *cell)
        den += weight
    return num / den


def excluding_bounds(share: float = NECESSITY_SHARE) -> dict[int, tuple[float, float]]:
    """N's control: the valid bounds with every :data:`NECESSITY_NODES` upper bound lowered.

    Each lowered bound lies :data:`NECESSITY_SHARE` of the way from the arm-1 at-risk mean of
    the true iterated mean (horizon 5) to its largest value over ``W``.
    """
    bounds = dict(study.MEAN_BOUNDS)
    for t in NECESSITY_NODES:
        low = at_risk_mean(t, 1)
        high = max(law.iterated_mean(t, 5, 1, *cell) for cell in law.CELLS)
        bounds[t] = (0.0, low + share * (high - low))
    return bounds


NECESSITY_BOUNDS = excluding_bounds()


def necessity_learners() -> dict[str, Any]:
    """N's learners: a saturated outcome regression and an intercept-only treatment mechanism.

    The outcome regression is per arm on ``W1, W2``, so the cell mean is ``trt*W1*W2``.  The
    censoring mechanism keeps the declared pooled logistic regression.
    """
    return {
        "outcome_learner": CellMeans(),
        "pseudo_learner": CellMeans(),
        "treatment_learner": MarginalProbabilities(),
    }


NECESSITY = property_verdicts.Necessity(
    family="bound_necessity",
    labels=("f5_arm1",),
    arms=("valid_bounds", "excluding_bound"),
    column="bound_displacement",
    threshold=NECESSITY_DISPLACEMENT,
)


# ---------------------------------------------------------------------------- the grid


class DeclaredLaw:
    """The law a declared cell reads: the study's one law, with its exact truths.

    Parameters
    ----------
    stream : str
        The property stream the cell draws from.
    """

    def __init__(self, stream: str) -> None:
        self.stream = stream
        self.name = f"survival_node_bounds__{stream}"

    def truth(self) -> dict[str, float]:
        return dict(law.TRUTH)


def synthetic_scale(cell: PropertyCell) -> float:
    """The standard deviation of a synthetic cell's estimates for the property differential.

    A calibration positive carries an exact-EIF efficiency band, so its synthetic estimates
    spread at the efficient influence function's standard deviation over ``sqrt(n)``.  N's
    arms spread at the intercept-only curve's.
    """
    if cell.property == "bound_necessity":
        sd = law.efficiency_sd(F5_ARM1, intercept_only=True)
    else:
        sd = law.EFFICIENCY_SD[cell.estimand]
    return float(sd / np.sqrt(cell.n))


def synthetic_columns(cell: PropertyCell, rng: np.random.Generator) -> dict[str, Any]:
    """The extra columns of a synthetic row: the convergence flag and W's two indicators."""
    columns: dict[str, Any] = {CONVERGENCE_COLUMN: 1}
    if cell.property == "binding_witness":
        columns["exceeds_bound"] = int(rng.random() < 0.6)
        columns["estimate_moved"] = int(rng.random() < 0.65)
    return columns


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its role, estimand, size and replicate count.

    A derived control (``shrunken_se_control``, ``noise_control``) is a cell of its positive
    arm's draws, so it carries that arm's size and count.
    """
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, name: str, n: int, stream: str) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(stream),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=REPLICATES_OF[stream],
                seed=stream_seed(STUDY, "property_sample", stream, 0),
                role=role,
                estimand=name,
            )
        )

    for family, names in PROPERTY_CELLS.items():
        for cell in names:
            if family == "bound_necessity":
                add(
                    family,
                    cell,
                    "control" if cell.endswith("excluding_bound") else "positive",
                    F5_ARM1,
                    NECESSITY_N,
                    "bound_necessity",
                )
                continue
            label = cell.split("__")[0]
            model = label.split("_model")[0] + "_model"
            if family == "interval_calibration":
                role = "positive" if cell.endswith("correctly_specified") else "control"
            elif family == "binding_witness":
                role = "positive" if model in WITNESS_GATED else DIAGNOSTIC_ROLE
            else:
                role = DIAGNOSTIC_ROLE
            name = (
                {**CALIBRATION, **REPORTED}[label][1]
                if label in {**CALIBRATION, **REPORTED}
                else F5_ARM1
            )
            add(family, cell, role, name, CALIBRATION_N, "calibration")
    return tuple(cells)


#: The replicate count of each stream.
REPLICATES_OF = {"calibration": CALIBRATION_REPLICATES, "bound_necessity": NECESSITY_REPLICATES}

#: Every fit set: ``(family, stream, n, replicates)``.
FIT_SETS: tuple[tuple[str, str, int, int], ...] = (
    ("calibration", "calibration", CALIBRATION_N, CALIBRATION_REPLICATES),
    ("bound_necessity", "bound_necessity", NECESSITY_N, NECESSITY_REPLICATES),
)

_Payload = tuple[str, str, int, int, int]
_Key = tuple[_Payload]


def _seed(stream: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", stream, replicate)


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, stream, replicate, n, _ = key[0]
    return law.draw(n, _seed(stream, replicate))


def cells_of(key: _Key) -> list[str]:
    """The ``family/cell`` names the fit set's rows and derived controls fill."""
    family = key[0][0]
    if family == "bound_necessity":
        return [f"bound_necessity/{cell}" for cell in PROPERTY_CELLS["bound_necessity"]]
    names = []
    for label in CALIBRATION_LABELS:
        base = f"interval_calibration/{label}__correctly_specified"
        names += [base, *DERIVED_CELLS[base]]
    for name in ("reported_calibration", "binding_witness", "unbounded_fit"):
        names += [f"{name}/{cell}" for cell in PROPERTY_CELLS[name]]
    return names


def cell_of(key: _Key) -> str:
    return f"{key[0][0]}/{key[0][1]}"


def replicate_of(key: _Key) -> int:
    return key[0][2]


def _converged(model: str, result: Any) -> bool:
    if model == study.HAZARD_MODEL:
        return all(fit.hazard.converged for fit in result.fits.values())
    return all(
        step.fluctuation is None or step.fluctuation.converged
        for fit in result.fits.values()
        for step in fit.steps
    )


def _row(
    family: str,
    cell: str,
    role: str,
    name: str,
    result: Any,
    model: str,
    *,
    replicate: int,
    n: int,
    requested: int,
) -> dict[str, Any]:
    row = replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=law.TRUTH[name],
        estimate=result[name],
        alpha=STUDY.margins.alpha,
    )
    row["converged"] = int(_converged(model, result))
    return row


def fit_drawn(payload: _Payload, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of one key on its drawn sample.

    Every row records whether the fit it reads converged (``converged``).  An unconverged fit
    is data, not a raised error: :func:`convergence_verdicts` fails a cell where more than
    :data:`UNCONVERGED_SHARE` of the replicates did not converge.
    """
    family, _, replicate, n, requested = payload
    common = {"replicate": replicate, "n": n, "requested": requested}
    mean = study.MEAN_MODEL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "bound_necessity":
            positive = study.fit(mean, frame, **necessity_learners())
            control = study.fit(mean, frame, bounds=NECESSITY_BOUNDS, **necessity_learners())
            return [
                _row(
                    family, "f5_arm1__valid_bounds", "positive", F5_ARM1, positive, mean, **common
                ),
                _row(
                    family, "f5_arm1__excluding_bound", "control", F5_ARM1, control, mean, **common
                ),
            ]
        rows: list[dict[str, Any]] = []
        for model in study.MODELS:
            bounds = study.declared_bounds(model)
            bounded = study.fit(model, frame)
            free = study.fit(model, frame, bounds=None)
            for label, (owner, name) in CALIBRATION.items():
                if owner != model:
                    continue
                rows.append(
                    _row(
                        "interval_calibration",
                        f"{label}__correctly_specified",
                        "positive",
                        name,
                        bounded,
                        model,
                        **common,
                    )
                )
                rows.append(
                    _row(
                        "unbounded_fit",
                        f"{label}__unbounded",
                        DIAGNOSTIC_ROLE,
                        name,
                        free,
                        model,
                        **common,
                    )
                )
            for label, (owner, name) in REPORTED.items():
                if owner == model:
                    rows.append(
                        _row(
                            "reported_calibration",
                            f"{label}__reported",
                            DIAGNOSTIC_ROLE,
                            name,
                            bounded,
                            model,
                            **common,
                        )
                    )
            exceeds = (
                study.mean_exceedance(free, bounds)
                if model == mean
                else study.hazard_exceedance(free, frame, bounds)
            )
            moved = max(abs(bounded.psi(k) - free.psi(k)) for k in (F5_ARM1, RD5))
            witness = _row(
                "binding_witness",
                f"{model}__binding",
                "positive" if model in WITNESS_GATED else DIAGNOSTIC_ROLE,
                F5_ARM1,
                bounded,
                model,
                **common,
            )
            witness["exceeds_bound"] = int(exceeds)
            witness["estimate_moved"] = int(moved > WITNESS_MOVEMENT)
            rows.append(witness)
        return rows


def _fit_key(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_drawn(key[0], frame)


def _payloads(budget: int | None, start: int = 0) -> list[_Key]:
    out: list[_Key] = []
    for family, stream, n, replicates in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(((family, stream, r, n, requested),) for r in range(start, start + requested))
    return out


#: The draw-level statistic the accounting reports for each lost replicate.
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = fit_properties(
        _payloads(budget),
        _fit_key,
        cells_of=cells_of,
        cell_of=cell_of,
        replicate_of=replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    columns = [*REPLICATE_COLUMNS, *WITNESS_COLUMNS, CONVERGENCE_COLUMN]
    return rows.reindex(columns=columns).sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts


def witness_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """W: both shares at least :data:`WITNESS_SHARE` on every gated model."""
    family = summary["property"] == "binding_witness"
    passes = []
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == "binding_witness") & (rows["cell"] == cell)]
        exceeds = float(group["exceeds_bound"].mean())
        moved = float(group["estimate_moved"].mean())
        summary.loc[index, "exceeds_share"] = exceeds
        summary.loc[index, "moved_share"] = moved
        passed = exceeds >= WITNESS_SHARE and moved >= WITNESS_SHARE
        if summary.loc[index, "role"] == DIAGNOSTIC_ROLE:
            passed = True
        else:
            passes.append(passed)
        summary.loc[index, "passed"] = bool(passed)
    summary.loc[family, "property_passed"] = bool(passes) and all(passes)


def convergence_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """The failure rule: a gated cell with more than :data:`UNCONVERGED_SHARE` unconverged fails.

    Every cell publishes its share.  A derived control's rows copy its positive arm's flags,
    so it reads that arm's share.  A gated cell over the share fails its own verdict, and the
    joint verdict of its group (:data:`CONVERGENCE_GROUPS`) fails with it.
    """
    for index in summary.index:
        group = rows.loc[
            (rows["property"] == summary.loc[index, "property"])
            & (rows["cell"] == summary.loc[index, "cell"])
        ]
        flags = group[CONVERGENCE_COLUMN].dropna()
        share = float(1.0 - flags.mean()) if len(flags) else 0.0
        summary.loc[index, "unconverged_share"] = share
    for family, joint in CONVERGENCE_GROUPS.items():
        member = summary["property"] == family
        failed = (
            member
            & (summary["role"] != DIAGNOSTIC_ROLE)
            & (summary["unconverged_share"] > UNCONVERGED_SHARE)
        )
        if not failed.any():
            continue
        summary.loc[failed, "passed"] = False
        summary.loc[failed if joint == "row" else member, "property_passed"] = False


def reported_families(summary: pd.DataFrame) -> None:
    """The diagnostic families state no verdict; C3 publishes both fits' RMSE."""
    rmse = np.sqrt(summary["bias"] ** 2 + summary["empirical_se"] ** 2)
    for label in UNBOUNDED:
        free = (summary["property"] == "unbounded_fit") & (summary["cell"] == f"{label}__unbounded")
        bounded = (summary["property"] == "interval_calibration") & (
            summary["cell"] == f"{label}__correctly_specified"
        )
        summary.loc[free, "rmse"] = rmse[free]
        summary.loc[free, "bounded_rmse"] = (
            float(rmse[bounded].iloc[0]) if bounded.any() else np.nan
        )
    for name in ("reported_calibration", "unbounded_fit"):
        mask = summary["property"] == name
        summary.loc[mask, "passed"] = True
        summary.loc[mask, "property_passed"] = True


SUMMARY_COLUMNS = (
    "bound_displacement",
    "exceeds_share",
    "moved_share",
    "rmse",
    "bounded_rmse",
    "unconverged_share",
)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=SUMMARY_COLUMNS,
        rate_labels=(),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(summary, rows, NECESSITY)
    witness_verdicts(summary, rows)
    reported_families(summary)
    convergence_verdicts(summary, rows)
    return finish(summary, rates)


def _share_of(column: str) -> Callable[[pd.DataFrame], np.ndarray]:
    """The per-replicate indicator one W share averages."""

    def value(group: pd.DataFrame) -> np.ndarray:
        return group[column].to_numpy(dtype=float)

    return value


def _witness_rule() -> property_failures.FamilyRule:
    """W: S10 floors on the two shares of each gated cell; joint over the family."""
    gated = property_failures.roles("positive")
    return property_failures.FamilyRule(
        families=frozenset({"binding_witness"}),
        shared="witness_verdicts",
        clauses=tuple(
            property_failures.extra_mean_clause(
                f"{column.replace('_', ' ')} share",
                value=_share_of(column),
                summary_column=summary_column,
                side="floor",
                limit=WITNESS_SHARE,
                applies=gated,
                support=(0.0, 1.0),
            )
            for column, summary_column in (
                ("exceeds_bound", "exceeds_share"),
                ("estimate_moved", "moved_share"),
            )
        ),
        group="family",
    )


def _unconverged(group: pd.DataFrame) -> np.ndarray:
    """The per-replicate indicator the unconverged share averages."""
    return 1.0 - group[CONVERGENCE_COLUMN].to_numpy(dtype=float)


def _with_convergence(rule: property_failures.FamilyRule) -> property_failures.FamilyRule:
    """``rule`` with the failure rule of :func:`convergence_verdicts` as one more clause.

    S10, a cap on the unconverged share of each gated cell, with a lost replicate filled as
    unconverged.  The clause feeds the cell's own verdict, so the rule's group carries it to
    the joint verdict.
    """
    assert rule.families <= set(CONVERGENCE_GROUPS)
    assert all(CONVERGENCE_GROUPS[family] == rule.group for family in rule.families)
    clause = property_failures.extra_mean_clause(
        "unconverged share",
        value=_unconverged,
        summary_column="unconverged_share",
        side="cap",
        limit=UNCONVERGED_SHARE,
        applies=property_failures.roles("positive", "control"),
        support=(0.0, 1.0),
    )
    return replace(rule, clauses=(*rule.clauses, clause))


#: The failure accounting's declared rules.  Each gated family's rule carries the convergence
#: clause (:func:`_with_convergence`).
PROPERTY_RULES = (
    _with_convergence(property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND)),
    *(_with_convergence(rule) for rule in property_verdicts.necessity_rules(NECESSITY)),
    _with_convergence(_witness_rule()),
)


__all__ = [
    "FIT_SETS",
    "NECESSITY",
    "PROPERTY_CELLS",
    "excluding_bounds",
    "generate_property_rows",
    "summarize_properties",
]


def smoke_keys(draws: int, start: int) -> list[_Key]:
    """Keys of a smoke on replicate ids ``start`` to ``start + draws - 1`` of every fit set."""
    return _payloads(draws, start)


def fit_keys(keys: Sequence[_Key]) -> list[list[dict[str, Any]]]:
    """Fit keys in this process, for a smoke outside the driver."""
    return [fit_drawn(key[0], _draw(key)) for key in keys]
