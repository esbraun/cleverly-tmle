r"""F5: the terminal experiment of the ``DRTMLE`` investigation.

``docs/roadmap.md``'s **F5** is the last statistical experiment in this investigation, and the
plan ends in a promotion or a stop rather than in another row.  F4 landed and returned a null --
no construction factor localizes the remaining calibration shortfall -- so what is left is the
narrow question this module is the instrument for:

    Can correctly or feasibly estimated reduced regressions close the remaining calibration
    gap, and does the answer depend on pooled against nested cross-fitting?

**One experiment in two frozen phases, and F8 is the second of them.**  Phase 1 is a selection
cohort over the fixed arm matrix, nominating at most one *feasible* configuration; phase 2 is
the untouched confirmation, carrying every one of F8's ten clauses.  The confirmation's seeds,
replicate counts, margins, exclusions and decision rules are committed **before phase 1
begins**, so the only value that crosses between the phases is the identity of the nomination.

Three things this module is built to get right, each because F4's run found the failure mode
---------------------------------------------------------------------------------------------

**The verdict rule is a partition and not a ladder of clauses.**  F4's ``_verdict`` tested
``moved`` before ``flat``, so an interval lying wholly inside its own negligible margin was
labelled a localization; and it was handed ``float("inf")`` as the band on every column but one,
which made the third verdict unreachable in five of six columns.  :func:`verdict` here is a
four-way partition on a **per-column** band that is finite and strictly positive by
construction, and :func:`format_rule_table` renders the rule so the prose and the predicate
cannot come apart -- ``tests/unit/test_drtmle_f5.py`` compares the two byte for byte.

**Six arms run of the matrix's eight, and each dropped cell has its own recorded reason.**  The
roadmap crosses four reduced learners with two cross-fitting constructions.  ``boost-pooled`` is
dropped on measured cost against what it could decide -- it is *diagnostic only, not nominable*
by the roadmap's own fence, and the pilot measured it at ``344 s`` a fit at ``n = 600`` against
the baseline's ``5.3 s`` -- see :data:`ARMS_DROPPED`, which also states what that costs.  The
other -- the ceiling at ``reduced_crossfit="nested"`` -- is a **null by construction**:
``reduced_crossfit`` acts only inside ``fit_reduced``, through
``NuisanceEstimates.inner``, on the reduced regressions' *training* rows, and
:class:`~benchmarks.drtmle_reference.ReferenceReductionDRTMLE` replaces the reduced set wholesale
in ``_nuisances`` and replaces ``spec.refit`` wholesale in ``_reduction``, so ``fit_reduced``
never produces an array the fit reads.  Both keywords therefore return bit-identical arrays,
and running the cell would spend a share of the phase-1 budget measuring a zero.
:func:`ceiling_crossfit_reading` answers it on the committed trace fixture instead, exactly and
deterministically -- which is the precedent F4 set when it moved its truncation contrast off a
cohort and onto two frozen fixtures, on the reasoning that *a declared stress design that cannot
be stressed is worse than no stress design: it spends fits and reads as evidence*.

**The ceiling's two error sources are measured by two different instruments**, because
conflating them is the way this arm gets over-read:

* its **numerical** error -- the reference's own Monte Carlo -- is the across-scramble spread of
  ``psi`` over independent reference randomisations (:func:`ceiling_adequacy`), read against the
  smallest decision margin in the frozen rule.  It is **not** a coarse/fine refinement pair;
  that statistic is withdrawn and F5 may not inherit a retracted statistic as its fidelity gate;
* its **smoothing** bias at tier 2 is bounded by nothing this module can run.  ``held_out_risk``
  estimates ``C + ||m - f||^2_w`` with ``C`` common to every candidate, so the reference gate
  observes a difference and never an absolute -- which is F6's open question, and F6 runs beside
  F5 rather than before it.

So the arm is reported as a **ceiling estimate** (:data:`CEILING_LABEL`) and never as an oracle
unless F6's absolute-adequacy route lands.  On the exact law it *is* an oracle, and
``--phase exact-law`` is the anchor that separates *the construction is right* from *the
smoothing at tier 2 is adequate*.

Usage
-----

.. code-block:: bash

    python -m benchmarks.drtmle_f5 --phase pilot --draws 2 --jobs 8      # timing only
    python -m benchmarks.drtmle_f5 --phase prereg --out evidence/f5-terminal
    python -m benchmarks.drtmle_f5 --phase select --cohort selection --sizes 600
    python -m benchmarks.drtmle_f5 --phase nominate --out evidence/f5-terminal
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

if __package__ in {None, ""}:  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import (
    drtmle_construction,
    drtmle_reference,
    drtmle_remainder,
    drtmle_stress,
    drtmle_tier2,
)

from cleverly import DRTMLE
from cleverly.estimators.base import format_table
from cleverly.learners import has_lightgbm
from cleverly.learners.library import _boost, _gam, _mean

__all__ = [
    "ARMS",
    "ARMS_DROPPED",
    "CEILING_LABEL",
    "COLUMNS",
    "SIZE_DRAWS",
    "Arm",
    "Column",
    "ContrastRow",
    "FitRow",
    "arm_estimator",
    "ceiling_crossfit_reading",
    "cohort_seeds",
    "component_risks",
    "contrast_rows",
    "default_jobs",
    "format_rule_table",
    "frozen_rule",
    "nominate",
    "one_draw",
    "prereg",
    "reduced_library",
    "refuse_dead_gates",
    "refuse_on_fallback",
    "resolved_implementations",
    "seeds_for",
    "validate_prereg",
    "verdict",
    "write_prereg",
]

# ------------------------------------------------------------------ the frozen design
#
# Every constant a verdict is read against, each with the reason it has the value it has.
# `frozen_rule()` serialises them and `validate_prereg` refuses a run whose committed manifest
# disagrees, so the record and the rule cannot come apart.

#: The two misspecification regimes phase 1 runs in, and they are C3c's and F4's.
CELLS = ("q-drift", "g-drift")

#: The applied stress cell, phase 2 only.  It discharges gate 2 clause 4 and stands in for F8
#: clause 3, whose paper reproduction is recorded **not feasible** -- see
#: :mod:`benchmarks.drtmle_stress` for the three citations that say why.
STRESS_CELL = drtmle_stress.CELLS[0]

#: Phase 1's sizes, F4's rather than C3c's three: every phase-1 outcome is a **paired**
#: difference within a draw, so a middle size buys a third reading of one comparison rather
#: than a rate.  Phase 2 takes three, because clause 4's trend is a rate.
SIZES = (600, 2_400)
CONFIRM_SIZES = (600, 1_200, 2_400)

#: Tier 2 -- both nuisances fitted.  A learner effect read under an injected nuisance would be
#: an effect on an estimator nobody runs.
TIER = 2

#: The primary estimand every verdict is read on.
PRIMARY_ESTIMAND = "ate"

#: The estimands every fit records.
ESTIMANDS = ("ate", "ey1", "ey0")

#: F4's committed draw counts, carried.  ``n = 600`` is declared in advance as powered for a
#: material move and not for equivalence, exactly as F4 declared it.
SIZE_DRAWS: dict[int, int] = {600: 24, 2400: 80}

#: Phase 2's replicate count **per batch**, and two independent batches.  At 500 the Monte
#: Carlo standard error of a coverage estimate is ``0.0097``, so the compatibility band is
#: ``+/- 0.019`` and a true ``0.92`` is separated from nominal.  C3c ran 250 and its own
#: specification records ``+/- 0.014`` there; the release criterion is the one place in this
#: study where the extra resolution buys a different conclusion rather than a tighter interval.
CONFIRM_REPLICATES = 500

#: The four verdicts.  ``unresolved`` is a recorded outcome and not a weak pass.
VERDICTS = ("equivalent", "improved", "worsened", "unresolved")

#: The bootstrap resample count for a paired interval, and the percentiles it is read at.
BOOTSTRAP = 2_000
INTERVAL = (2.5, 97.5)

#: A run is complete when it holds at least this share of its declared draws.
COMPLETENESS_FRACTION = 0.9

#: The score-check tolerance a fit's validity is read at.
VALIDITY_TOLERANCE = 1e-3

#: The quadrature rule ``P_0 D-hat`` is integrated on, and the scramble count per draw.
QUADRATURE_POINTS = 2_048
QUADRATURE_SCRAMBLES = 2

#: The ceiling arm's reference block.  E2R's count, and a reference sharing a scramble stream
#: with the evaluation block would make the reference's error and ``P_0 D-hat`` the same random
#: variable with a covariance nobody can sign -- so this block is its own stream.
REFERENCE_POINTS = 8_192

#: How many independent reference randomisations :func:`ceiling_adequacy` reads the numerical
#: error over.  Four, because the statistic is a spread and two replicates do not estimate one.
REFERENCE_SCRAMBLES = 4

#: The frozen ceiling rung, and it is chosen on E2R's **audit** rather than on taste.  E2R's
#: selection cohort picked ``spline(8)`` in ten of twelve (cell, size, regression) rows, but the
#: audit resolved exactly one rung-against-rung comparison and it went the other way: at
#: ``q-drift``, ``n = 2,400``, on ``q_r``, ``spline(16)`` measurably beat ``spline(8)``.  The
#: other three cells failed only the non-inferiority clause, which is *not shown equal* rather
#: than *shown worse*.  ``spline(16)`` is also E2's shipped rung and the one the study manifest
#: already names, so freezing it means this arm inherits E2's identity rather than inventing one.
CEILING_KNOTS = 16

#: The fewest Sobol **points** a reference block may carry, which is not the fewest rows.
#: :class:`~benchmarks.drtmle_reference.SplineProjection` budgets ``POINTS_PER_PARAMETER = 64``
#: rows per basis column and ``spline(16)`` has 19 of them, so it refuses a fit on fewer than
#: ``1,216`` rows -- and :math:`Q_r`'s fit is **masked to one arm** (``fit_mask``'s ``| A = a``).
#: A quadrature block interleaves the two arms one row each, so a block of ``P`` points offers
#: ``2P`` rows but only ``P`` of them to :math:`Q_r`.  The floor is therefore on points, not on
#: rows, and getting that backwards raises **mid-cohort** rather than at construction: measured,
#: at ``points=1024``, as "spline(16) ... would be fitted on 1024 rows".
#: **Rounded up to a power of two**, because ``DGP.quadrature`` refuses anything else -- a
#: coarser grid has to be a prefix of a finer one -- so the arithmetic floor of ``64 x 19 =
#: 1,216`` is not a reachable point count and ``2,048`` is the first one that is.  A guard
#: stated at the arithmetic floor would pass a value the grid then rejects several frames
#: deeper, which is the same "raises late rather than at construction" failure this constant
#: exists to prevent.
MINIMUM_REFERENCE_POINTS = (
    1
    << (
        drtmle_reference.POINTS_PER_PARAMETER
        * drtmle_reference.SplineProjection(CEILING_KNOTS).n_parameters
        - 1
    ).bit_length()
)

#: What the ceiling arm is called, in one place.  It is an **oracle** on the exact law and a
#: **ceiling estimate** at tier 2, and it is never promoted to the first by being the best
#: number available.  ``tests/unit/test_drtmle_f5.py`` asserts no tier-2 table says "oracle".
CEILING_LABEL = "ceiling estimate"

#: The seed-stream family.  ``90``--``92M`` are C3c's and E1b's, ``103``--``106M`` are E2R's and
#: ``110M`` is F4's, so F5 takes ``120M``: two studies sharing a stream would be the same rows
#: under two headings.
COHORT_SEED = 20260201
CONFIRM_SEED_A = 20260202
CONFIRM_SEED_B = 20260203
QUADRATURE_SEED = 120_000_000
REFERENCE_SEED = 121_000_000
SCORING_SEED = 122_000_000

#: The two phase-1 cohorts, in the order they are read.
COHORTS = ("selection", "audit")

#: A **third** child of the same seed sequence, reserved for the timing pilot and spent there.
#: ``SeedSequence.spawn`` gives child ``i`` the same state whatever ``n`` is, so reserving it
#: leaves the two cohorts byte-identical to what ``spawn(2)`` would have produced.
SIZING_STREAM = 2

#: The cohort label the pilot's rows carry.  :func:`contrast_rows` and :func:`nominate` **raise**
#: on it: the pilot is timing only, and sizing comes from F4's committed ``PILOT_PAIRED_SPREAD``
#: and nothing else.  Enforcing that structurally is cheaper than remembering it.
SIZING_COHORT = "sizing"


# ------------------------------------------------------------------ the arms


@dataclass(frozen=True)
class Arm:
    """One configuration of the reduced regressions, and what it is allowed to become."""

    learner: str
    crossfit: str
    role: str
    nominable: bool
    why: str


#: The six arms that run, of the roadmap matrix's eight.  **Two cells are dropped and each for
#: its own recorded reason, both before the freeze and neither after seeing an estimate.**
#:
#: ``ceiling-nested`` is a **null by construction** -- ``reduced_crossfit`` cannot reach an arm
#: whose reduced set is replaced wholesale -- and :func:`ceiling_crossfit_reading` answers it
#: exactly instead.
#:
#: ``boost-pooled`` is dropped **on measured cost against what it could decide**.  The roadmap
#: marks it *diagnostic only, not nominable*: A1b's argument carries the pooled construction on
#: a one-dimensional bounded-variation ball or a fixed-dimension sieve, a boosted reduction's
#: pooled design/target-continuity premise is not closed, and so ``boost`` reaches a production
#: branch under ``nested`` or not at all.  The timing pilot then measured it at ``344 s`` a fit
#: at ``n = 600`` against the baseline's ``5.3 s`` -- with the two boost arms together at 94% of
#: a draw's entire cost -- so this cell alone is most of a phase-1 budget spent on an arm no
#: branch of the terminal plan can read.
#:
#: **What that costs is stated rather than absorbed**: the cross-fitting axis stays identifiable
#: at ``glm`` and at ``gam``, and the learner-by-cross-fitting *interaction* is **not** available
#: at ``boost``.  The **cross-fitting column exists because F4 measured something on it** --
#: ``nested ~ cleverly`` moved the point estimate in seven readings of eight and reproduced in
#: ``q-drift``, on a secondary column that decides nothing -- so keeping the axis identifiable
#: somewhere is the point, and it is kept in two of the three learner rows.
ARMS_DROPPED: dict[str, str] = {
    "ceiling-nested": "a null by construction; ceiling_crossfit_reading() answers it exactly. "
    "Dropped BEFORE any inferential fit, on a structural fact about the code",
    "boost-pooled": "diagnostic-only by the roadmap's own fence and 344 s a fit at n=600, so it "
    "is most of a phase-1 budget spent on an arm no branch can read. Dropped BEFORE any "
    "inferential fit, on the timing pilot alone",
    "boost-nested": "withdrawn on cost AFTER 41 partial selection draws had been read, which is "
    "why this entry is worded differently from the two above. boost-nested was 77% of a draw "
    "and ~13 h of a ~17 h phase-1 budget. It is NOT withdrawn because those draws looked bad -- "
    "selecting an arm on its own outcome is what a preregistration exists to prevent -- and no "
    "reading of it is carried forward or reported as a result. The scope decision is that F5 "
    "asks whether DRTMLE is *constructed* correctly, and a boosted reduction is one candidate "
    "for that and not the question. What it costs: F5 makes no claim about boosted reductions "
    "at all, and its learner screen is a fixed-basis smoother against the shipped GLM against "
    "the ceiling",
}
ARMS: dict[str, Arm] = {
    "glm-pooled": Arm(
        learner="glm",
        crossfit="pooled",
        role="baseline",
        nominable=False,
        why="the shipped configuration, and C3c's",
    ),
    "glm-nested": Arm(
        learner="glm",
        crossfit="nested",
        role="comparator",
        nominable=False,
        why="the construction comparator: the shipped learner at the reference cross-fitting",
    ),
    "gam-pooled": Arm(
        learner="gam",
        crossfit="pooled",
        role="candidate",
        nominable=True,
        why="a fixed-basis smoother, whose pooled premise A1b's argument does carry",
    ),
    "gam-nested": Arm(
        learner="gam",
        crossfit="nested",
        role="candidate",
        nominable=True,
        why="the same smoother at the reference cross-fitting",
    ),
    "ceiling": Arm(
        learner="ceiling",
        crossfit="pooled",
        role="ceiling",
        nominable=False,
        why="a ceiling measures an attainable bound and is not a procedure a caller can run; "
        "its nested cell is bit-identical and is read by ceiling_crossfit_reading()",
    ),
}

#: The baseline every contrast is read against.
BASELINE_ARM = "glm-pooled"


def reduced_library(learner: str, random_state: int) -> dict[str, Any]:
    r"""The two task-specific slots one arm's reduced regressions are fitted with.

    **Two slots and not one**, because ``DRTMLE._fit_reduced`` resolves
    ``reduced_outcome_learner`` at ``task="regression"`` -- which covers :math:`Q_r` *and*
    :math:`g_{r,2}` -- and ``reduced_treatment_learner`` at ``task="classification"``, which
    covers :math:`g_{r,1}` alone.  A single object would be resolved at one task and a
    classifier cannot serve :math:`Q_r`, whose target is a signed residual.

    **``mean`` is kept in every library, and that is a decision rather than a convenience.**
    The shipped baseline is the ``"glm"`` preset, which is ``mean + glm`` -- a two-candidate
    ensemble.  A single-candidate ``gam`` arm would move the function class **and** the ensemble
    shape, which is two factors, and F4's whole matrix exists because a contrast that moves two
    things cannot say which of them moved the number.  Whether an arm then collapses onto
    ``mean`` is a **recorded column** (``flex_weight_min`` on every :class:`FitRow`) rather than
    an inference.

    ``"gam"`` and ``"boost"`` are not preset names -- the presets are ``glm``, ``fast``,
    ``default`` and ``rich`` -- so both are built from the private factories and handed over as a
    named ``[(name, estimator)]`` library, which still goes through ``SuperLearner`` and so still
    produces the per-fold diagnostics the mechanism columns read.
    """
    if learner == "glm":
        return {"reduced_outcome_learner": "glm", "reduced_treatment_learner": "glm"}
    if learner not in {"gam", "boost"}:
        raise ValueError(f"unknown reduced learner {learner!r}")

    def build(task: str) -> Any:
        if learner == "gam":
            return _gam(task)
        # Thread counts are pinned by F5's row, and `_boost` sets none: LightGBM's `n_jobs`
        # defaults to -1, which would put a whole machine's worth of threads inside a worker
        # that is already one of ten.
        return _boost(task, random_state).set_params(n_jobs=1)

    return {
        "reduced_outcome_learner": [("mean", _mean("regression")), (learner, build("regression"))],
        "reduced_treatment_learner": [
            ("mean", _mean("classification")),
            (learner, build("classification")),
        ],
    }


def resolved_implementations(random_state: int = 0) -> dict[str, dict[str, str]]:
    """What every arm's flexible candidate resolves to, fitting nothing.

    Recorded in the manifest and re-derived at every run, so that a study dispatched on a box
    where an optional extra is missing cannot report two different function classes under one
    arm name -- which would make the entropy row of the supported contract ambiguous about which
    was fitted.
    """
    out: dict[str, dict[str, str]] = {}
    for name, arm in ARMS.items():
        if arm.learner in {"glm", "ceiling"}:
            out[name] = {
                "regression": arm.learner,
                "classification": arm.learner,
            }
            continue
        library = reduced_library(arm.learner, random_state)
        out[name] = {
            "regression": type(library["reduced_outcome_learner"][1][1]).__name__,
            "classification": type(library["reduced_treatment_learner"][1][1]).__name__,
        }
    return out


def refuse_on_fallback() -> list[str]:
    """Every reason a boosted fit here would not be the one declared.  Empty is the pass.

    F5's row requires LightGBM and refuses the run if ``library._boost``'s
    ``HistGradientBoosting`` fallback is what got fitted.  ``_boost`` returns the fallback under
    the **same** ``"boost"`` name, so nothing downstream would notice: the diagnostics carry the
    library entry's name and not its class.

    **This outlives the boosted arms, and deliberately.**  Both boost cells are withdrawn from
    the arm matrix, so no *reduced* regression fits LightGBM any more -- but phase 2's applied
    stress cell fits its **primary** nuisances with the ``"fast"`` preset, which contains
    ``boost``.  A box without the extra would quietly fit a different function class there under
    the same preset name, which is the same ambiguity for the same reason.
    """
    complaints: list[str] = []
    if not has_lightgbm():
        complaints.append(
            "boost: cleverly.learners.has_lightgbm() is False -- install the `boost` extra "
            "(`pip install -e '.[dev]'`); this study declares LightGBM and refuses the fallback"
        )
    for task, expected in (("regression", "LGBMRegressor"), ("classification", "LGBMClassifier")):
        got = type(_boost(task, 0)).__name__
        if got != expected:
            complaints.append(
                f"boost/{task}: library._boost resolved to {got}, not {expected} -- the "
                "HistGradientBoosting fallback is not the arm this study declared, and two "
                "function classes under one arm name make the contract's entropy row ambiguous"
            )
    return complaints


def arm_estimator(
    arm: str,
    settings: Mapping[str, Any],
    *,
    random_state: int,
    evaluation: Any = None,
    dgp: Any = None,
    window: Any = None,
    row_weights: Any = None,
    **kwargs: Any,
) -> DRTMLE:
    """The estimator for one arm, at the shared settings for its cell.

    The ceiling arm is a different class rather than a keyword, because it replaces both the
    initial reduced set and the refit closure -- overriding only the second leaves one fitted
    set in the fit, which is the half-substitution
    :class:`~benchmarks.drtmle_reference.ReferenceReductionDRTMLE` rejects by name.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
    spec = ARMS[arm]
    shared: dict[str, Any] = {
        **settings,
        "random_state": random_state,
        "evaluation": evaluation,
        **kwargs,
    }
    if spec.learner == "ceiling":
        if dgp is None or window is None or row_weights is None:
            raise ValueError(
                "the ceiling arm needs dgp=, window= and row_weights=: its reference is fitted "
                "on a Sobol block of the law's own grid, and the weights must be that block's"
            )
        return drtmle_reference.ReferenceReductionDRTMLE(
            dgp=dgp,
            reference=drtmle_reference.SplineProjection(CEILING_KNOTS),
            window=window,
            row_weights=row_weights,
            reduced_crossfit=spec.crossfit,
            **shared,
            **reduced_library("glm", random_state),
        )
    return DRTMLE(
        reduced_crossfit=spec.crossfit,
        **shared,
        **reduced_library(spec.learner, random_state),
    )


# ------------------------------------------------------------------ the columns and the rule


@dataclass(frozen=True)
class Column:
    """One reported column, and the band a verdict on it is read against.

    ``band`` is finite and strictly positive **by construction** -- checked at import and again
    in :func:`validate_prereg`.  That is F4's second rule defect made unreachable rather than
    documented: it passed ``float("inf")`` for every non-primary column, which made
    ``unresolved`` impossible to reach and so made five of six columns report a verdict they
    could not have failed to report.
    """

    name: str
    group: str
    band: float
    relative: bool
    #: ``-1`` lower is better, ``+1`` higher is better, ``0`` no direction (cost, diagnosis).
    orientation: int
    #: ``"paired"`` -- a per-draw difference; ``"cohort"`` -- a set functional, bootstrapped over
    #: draw indices with both arms recomputed on the same resample.
    statistic: str
    #: Whether any nomination clause reads this column.
    gates: bool
    why: str


#: Every column, with the band tied to a tolerable change in the reported quantity and the
#: anchor it is a fraction or a multiple of.
#:
#: **Three of these could not be anchored absolutely, and that is a finding about the record
#: rather than a shortcut.**  C3c committed no RMSE for ``ate`` in any cell, at any size, in
#: either batch; empirical SD and mean reported ``se`` only for ``q-drift``, ``n = 2,400``,
#: batch A; root-``n`` bias only for ``q-drift``, batch A.  Inventing an absolute band for those
#: would be an invented threshold wearing a citation, so they are declared **relative** -- a
#: tenth of the baseline arm's own realized value in that cell and size -- before the first fit.
COLUMNS: tuple[Column, ...] = (
    Column(
        "root_n_remaining",
        "theorem",
        0.125,
        False,
        -1,
        "paired",
        True,
        "0.10 of C3c's q-drift reading (1.25); F4's NEGLIGIBLE_EFFECT x C3C_REMAINING_QDRIFT "
        "carried verbatim so the two studies share a scale. g-drift takes the same fraction of "
        "its own committed level, 4.13, since one band across cells whose levels differ "
        "threefold is a band for one of them -- see BAND_BY_CELL",
    ),
    Column(
        "score_8",
        "theorem",
        1e-4,
        False,
        -1,
        "paired",
        False,
        "an order below VALIDITY_TOLERANCE, the tolerance a fit's validity is read at",
    ),
    Column(
        "score_9",
        "theorem",
        1e-4,
        False,
        -1,
        "paired",
        False,
        "an order below VALIDITY_TOLERANCE",
    ),
    Column(
        "score_10",
        "theorem",
        1e-4,
        False,
        -1,
        "paired",
        False,
        "an order below VALIDITY_TOLERANCE",
    ),
    Column(
        "score_failures",
        "theorem",
        0.02,
        False,
        -1,
        "paired",
        True,
        "gate 2 clause 2's committed invalid-fit threshold, on the per-fit rate scale. F4 gave "
        "this column no band at all, which is the defect this repairs",
    ),
    Column(
        "bound_active",
        "theorem",
        0.01,
        False,
        0,
        "cohort",
        False,
        "C3c's realized bound-active share ranges 0.012 to 0.088; 0.01 is the smallest move "
        "that could change which section 7 scope a cell sits in",
    ),
    Column(
        "abs_error",
        "estimator",
        0.10,
        True,
        -1,
        "paired",
        False,
        "relative: no committed absolute exists for the absolute ATE error in these cells",
    ),
    Column(
        "root_n_bias",
        "estimator",
        0.10,
        True,
        -1,
        "cohort",
        False,
        "relative: C3c committed root-n bias for q-drift batch A only",
    ),
    Column(
        "rmse",
        "estimator",
        0.10,
        True,
        -1,
        "cohort",
        False,
        "relative: C3c committed no RMSE anywhere -- it appears in the record only as a column "
        "F5 is required to report",
    ),
    Column(
        "empirical_sd",
        "estimator",
        0.10,
        True,
        0,
        "cohort",
        False,
        "relative: committed for q-drift n=2,400 batch A only",
    ),
    Column(
        "se_ratio",
        "estimator",
        0.10,
        False,
        -1,
        "cohort",
        False,
        "gate 1 clause 5's committed [0.90, 1.10], which q-drift passed at 0.903 and g-drift "
        "failed at 1.157",
    ),
    Column(
        "abs_coverage_gap",
        "estimator",
        0.05,
        False,
        -1,
        "cohort",
        True,
        "gate 2 clause 1's committed shortfall margin, the only committed coverage margin in "
        "the record. Phase 1's draw counts cannot resolve smaller, so this column reads "
        "unresolved in most phase-1 cells by design -- it is a veto, and unresolved fires none",
    ),
    Column(
        "risk_qr",
        "mechanism",
        0.10,
        True,
        -1,
        "paired",
        False,
        "diagnosis, not proof of consistency: a held-out risk difference ranks candidates and "
        "never bounds ||m - f||^2_w",
    ),
    Column(
        "risk_gr1",
        "mechanism",
        0.10,
        True,
        -1,
        "paired",
        False,
        "diagnosis, not proof of consistency",
    ),
    Column(
        "risk_gr2",
        "mechanism",
        0.10,
        True,
        -1,
        "paired",
        False,
        "diagnosis, not proof of consistency",
    ),
    Column(
        "risk_h3",
        "mechanism",
        0.10,
        True,
        -1,
        "paired",
        True,
        "the composite q_r/g the fit actually divides by; a veto only",
    ),
    Column(
        "risk_h2",
        "mechanism",
        0.10,
        True,
        -1,
        "paired",
        True,
        "the composite g_r2/g_r1 the fit actually divides by; a veto only",
    ),
    Column(
        "seconds",
        "cost",
        1.0,
        True,
        0,
        "cohort",
        False,
        "cost; measured serially by --phase cost, since a runtime taken under ten-way "
        "contention is a measurement of the scheduler",
    ),
)

#: ``root_n_remaining``'s band is per cell, because the column sits at ``1.25`` in ``q-drift``
#: and ``4.13`` in ``g-drift`` and one absolute band across both would be a band for one of
#: them.  The **fraction** is one number -- F4's ``0.10`` -- applied to each cell's own
#: committed level, which is what keeps the two studies on one scale.
C3C_REMAINING = {"q-drift": 1.25, "g-drift": 4.13}
NEGLIGIBLE_FRACTION = 0.10
BAND_BY_CELL = {
    "root_n_remaining": {
        cell: round(NEGLIGIBLE_FRACTION * level, 4) for cell, level in C3C_REMAINING.items()
    }
}

#: **Identity failures are not a banded column.**  C3c recorded **zero** across all 6,000 fits,
#: so the study has an exact answer for this quantity and turning it into a paired difference
#: with a tolerance would make the one thing that is certain into something statistical.  Any
#: nonzero count on any arm is a veto, and it is a stop-immediately condition besides.
IDENTITY_FAILURES_ARE_EXACT = True

#: How small a flexible candidate's **mean** SuperLearner weight may get before the arm is the
#: baseline under another name, and in what share of *fits* that mean must clear it.
#:
#: **Corrected before the cohort was read**, and the correction is on the statistic and not on
#: the threshold.  As first frozen the clause read ``flex_weight_min`` -- the minimum over three
#: reduced regressions and five folds -- and asked *that* to clear ``0.05``.  A single fold in
#: which an ensemble puts no weight on the flexible candidate is ordinary, so the minimum sits
#: at zero for almost any arm: on the first 41 partial draws it read ``0.0000`` at every
#: quantile including the maximum for the boosted arm and at the median for both spline arms.
#: No arm could have passed, so the study would have returned "no nomination" as an artefact of
#: its own predicate -- F4's rule defect rebuilt under a new name, which F5's row exists to
#: prevent.  ``0.05`` and ``0.90`` are unchanged; only the quantity they are applied to moved.
FLEX_WEIGHT_FLOOR = 0.05
FLEX_WEIGHT_SHARE = 0.90

COLUMN_BY_NAME = {column.name: column for column in COLUMNS}


def band_for(column: Column, cell: str, baseline: float | None = None) -> float:
    """The band this column is read at, in this cell, on this scale.

    Three cases and each is declared: a per-cell absolute (``root_n_remaining``), a relative band
    resolved against the baseline arm's own realized value, and a plain absolute.
    """
    if column.name in BAND_BY_CELL:
        return float(BAND_BY_CELL[column.name].get(cell, column.band))
    if column.relative:
        if baseline is None or not math.isfinite(baseline) or baseline == 0.0:
            return float("nan")
        return float(column.band * abs(baseline))
    return float(column.band)


def verdict(lower: float, upper: float, band: float, orientation: int) -> str:
    r"""A **non-overlapping** three-way partition, plus ``unresolved``, on one column's scale.

    At most one of the first three can fire, and none of them is unreachable:

    * ``equivalent`` needs ``-band <= lower`` **and** ``upper <= band``;
    * a beyond-verdict needs ``lower > band`` -- which forces ``upper >= lower > band`` and so
      contradicts ``upper <= band`` -- or ``upper < -band``, which forces
      ``lower <= upper < -band`` and contradicts ``-band <= lower``;
    * every declared band is finite and strictly positive, so ``[0, 0]`` is ``equivalent``,
      ``[2b, 3b]`` is beyond and ``[-2b, 2b]`` is ``unresolved``: all three are reachable for
      every column.

    ``orientation`` turns a *direction* into a *judgement* and nothing else: ``-1`` means lower
    is better, so an interval above the band is ``worsened``.  A column with orientation ``0``
    is reported and never gated, so its beyond-verdicts are named ``improved``/``worsened`` for
    table symmetry and read by no clause.

    This is F4's rule defect repaired rather than reordered.  F4 tested ``moved`` first, so
    ``[-1e-4, -1e-5]`` -- an interval lying wholly inside a ``+/- 0.125`` margin -- was labelled
    a localization.  Here that interval is ``equivalent`` and nothing else can fire.
    """
    if not (math.isfinite(lower) and math.isfinite(upper) and math.isfinite(band) and band > 0.0):
        return "unresolved"
    if -band <= lower and upper <= band:
        return "equivalent"
    if lower > band:
        return "worsened" if orientation < 0 else "improved"
    if upper < -band:
        return "improved" if orientation < 0 else "worsened"
    return "unresolved"


RULE_HEADERS = ("column", "group", "band", "scale", "better", "statistic", "gates")


def format_rule_table() -> str:
    """The frozen rule, rendered.

    ``docs/drtmle/terminal-experiment.md`` carries this table verbatim and
    ``tests/unit/test_drtmle_f5.py`` compares the two byte for byte, so the prose and the
    predicate cannot come apart -- which is the other half of F4's rule defect, where one line
    of the roadmap named two primary outcomes and the code banded one of them.
    """
    body = []
    for column in COLUMNS:
        if column.name in BAND_BY_CELL:
            band = ", ".join(
                f"{cell} {value:g}" for cell, value in BAND_BY_CELL[column.name].items()
            )
        else:
            band = f"{column.band:g}"
        body.append(
            (
                column.name,
                column.group,
                band,
                "relative" if column.relative else "absolute",
                {-1: "lower", 0: "--", 1: "higher"}[column.orientation],
                column.statistic,
                "yes" if column.gates else "no",
            )
        )
    return format_table(RULE_HEADERS, body)


def frozen_rule() -> dict[str, Any]:
    """Every constant a verdict is read against, as a dictionary a manifest can carry."""
    return {
        "band_by_cell": BAND_BY_CELL,
        "bootstrap": BOOTSTRAP,
        "ceiling_knots": CEILING_KNOTS,
        "ceiling_label": CEILING_LABEL,
        "columns": [asdict(column) for column in COLUMNS],
        "completeness_fraction": COMPLETENESS_FRACTION,
        "flex_weight_floor": FLEX_WEIGHT_FLOOR,
        "flex_weight_share": FLEX_WEIGHT_SHARE,
        "identity_failures_are_exact": IDENTITY_FAILURES_ARE_EXACT,
        "interval": list(INTERVAL),
        "negligible_fraction": NEGLIGIBLE_FRACTION,
        "primary_estimand": PRIMARY_ESTIMAND,
        "reference_scrambles": REFERENCE_SCRAMBLES,
        "smallest_decision_margin": smallest_decision_margin(),
        "validity_tolerance": VALIDITY_TOLERANCE,
        "verdicts": list(VERDICTS),
    }


def smallest_decision_margin() -> float:
    """The smallest absolute band any gating column is read at.

    This is what the ceiling arm's **numerical** error has to be small against -- F5's second
    ceiling rule -- so it is derived from the rule rather than chosen beside it.  Relative bands
    are excluded because they have no value until a baseline exists.
    """
    absolute = [
        band_for(column, cell)
        for column in COLUMNS
        if column.gates and not column.relative
        for cell in (*CELLS, STRESS_CELL)
    ]
    finite = [value for value in absolute if math.isfinite(value) and value > 0.0]
    return float(min(finite)) if finite else float("nan")


# ------------------------------------------------------------------ the seeds


def cohort_seeds(seed: int, draws: int) -> dict[str, list[tuple[int, int]]]:
    """One ``(data_seed, fold_seed)`` list per cohort, from **disjoint** streams.

    F4's device, reused rather than re-derived: ``spawn`` rather than a slice, because a slice is
    prefix-stable and raising one cohort's count would shift which draws the other took --
    exactly what the study manifest records C3c running into.
    """
    children = np.random.SeedSequence(seed).spawn(SIZING_STREAM + 1)
    out: dict[str, list[tuple[int, int]]] = {}
    for name, child in zip(COHORTS, children, strict=False):
        state = child.generate_state(2 * draws)
        out[name] = [
            (int(data), int(fold)) for data, fold in zip(state[:draws], state[draws:], strict=True)
        ]
    return out


def sizing_seeds(seed: int, draws: int) -> list[tuple[int, int]]:
    """The pilot's draws: the reserved third child, which no cohort reads."""
    child = np.random.SeedSequence(seed).spawn(SIZING_STREAM + 1)[SIZING_STREAM]
    state = child.generate_state(2 * draws)
    return [(int(a), int(b)) for a, b in zip(state[:draws], state[draws:], strict=True)]


def confirm_seeds(seed: int, replicates: int) -> list[tuple[int, int]]:
    """One batch of confirmation draws, from its own seed."""
    state = np.random.SeedSequence(seed).generate_state(2 * replicates)
    return [(int(a), int(b)) for a, b in zip(state[:replicates], state[replicates:], strict=True)]


def seeds_for(manifest: Mapping[str, Any], cohort: str, size: int) -> list[tuple[int, int]]:
    """The draws one ``(cohort, size)`` runs, as a **prefix** of the committed cohort."""
    declared = [(int(a), int(b)) for a, b in manifest["phase1"]["cohorts"][cohort]]
    return declared[: SIZE_DRAWS.get(size, len(declared))]


# ------------------------------------------------------------------ the rows


@dataclass(frozen=True)
class FitRow:
    """One arm on one draw, flat and JSON-serialisable so an artefact needs no schema to read."""

    cohort: str
    cell: str
    n: int
    data_seed: int
    fold_seed: int
    arm: str
    estimand: str
    psi: float
    truth: float
    #: ``|psi - truth|`` on this draw.  A field rather than a derivation at read time, because
    #: the paired columns are read off :class:`FitRow` by name and a column declared in
    #: :data:`COLUMNS` with no field behind it would raise mid-cohort rather than at import.
    abs_error: float
    std_error: float
    root_n_remaining: float
    score_8: float
    score_9: float
    score_10: float
    identity_failures: int
    score_failures: int
    valid: bool
    rounds: int
    closing: int
    exit_reason: str
    failure: str
    bound_active: bool
    initial_clip_share: float
    #: The flexible candidate's **mean** SuperLearner weight over folds and reduced regressions.
    #: An arm that collapsed onto ``mean`` is the baseline under another name, and this is what
    #: says so in the artefact rather than in an argument.  It is the mean and not the minimum
    #: because the minimum is zero for almost any arm -- see :func:`_flex_weights`.
    flex_weight_mean: float
    #: The minimum over the same fifteen values, kept as a **diagnostic beside** the mean and
    #: read by no clause.  It is what the nomination rule used to test, and it is retained so a
    #: reader can see the correction rather than take it on trust.
    flex_weight_min: float
    risk_qr: float
    risk_gr1: float
    risk_gr2: float
    risk_h3: float
    risk_h2: float
    #: The resolved implementation of the flexible candidate, per row, so a fallback that
    #: somehow got fitted is visible in the artefact and not only in a preflight.
    impl: str
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class ContrastRow:
    """One arm against the baseline, on one column, in one cell -- the grain a verdict is at."""

    cohort: str
    cell: str
    n: int
    arm: str
    role: str
    column: str
    group: str
    draws: int
    mean: float
    lower: float
    upper: float
    paired_sd: float
    band: float
    verdict: str


@dataclass(frozen=True)
class Payload:
    """Everything one draw needs, so a worker takes one picklable argument."""

    cohort: str
    cell: str
    n: int
    data_seed: int
    fold_seed: int
    arms: tuple[str, ...]
    quadrature_points: int
    reference_points: int


# ------------------------------------------------------------------ one draw


def _law_and_settings(cell: str, n: int) -> tuple[Any, dict[str, Any]]:
    """The law and the shared estimator settings for one cell, whichever module supplies it."""
    if cell == STRESS_CELL:
        return drtmle_stress.base_law(), dict(drtmle_stress.settings(cell, n))
    return drtmle_tier2.base_law(), dict(drtmle_tier2.settings(cell, n))


@dataclass(frozen=True)
class _Context:
    """One companion and everything read off it, so an arm takes the right one as a unit."""

    stack: Any
    windows: list[Any]
    truths: list[Any]
    scoring: np.ndarray
    reference_window: Any | None


def _context(payload: Payload, dgp: Any, *, with_reference: bool) -> _Context:
    """Build a companion and the windows, truths and scoring mask that go with it.

    **Two of these are built per draw when a ceiling arm runs, and that is the point.**  The
    companion is handed to every fit through ``evaluation=``, and a fit *predicts at every
    companion row* -- so the ceiling's 8,192-point reference block was being paid for by all six
    arms, not by the one that reads it.  Measured on a ``glm-pooled`` fit at ``n = 600``: 15.9 s
    against a lean companion of 12,288 rows and 23.2 s against the 28,672-row shape the cohort
    was running, and the effect is worse on an arm whose reduced regressions refit many times.

    Splitting them changes **no result**: the companion contributes to no fit, no fold and no
    score (``tests/unit/test_drtmle_companion.py`` pins that bit for bit), and the evaluation
    blocks are built at the same points and the same scrambles in both, so they are the same
    rows in the same order and the remainder is integrated on the same grid either way.  What
    changes is only how many rows the arms that never look at the reference block have to
    predict at.
    """
    stack, scoring_index, reference_index = _companion(payload, dgp, with_reference=with_reference)
    evaluation = [
        block
        for index, block in enumerate(stack.blocks)
        if index not in {scoring_index, reference_index}
    ]
    scoring = np.zeros(stack.weights.size, dtype=bool)
    window = stack.blocks[scoring_index].window
    scoring[window.start : window.stop] = True
    return _Context(
        stack=stack,
        windows=[block.window for block in evaluation],
        truths=[
            drtmle_remainder.truth_at(dgp, block.points, scramble=block.seed)
            for block in evaluation
        ],
        scoring=scoring,
        reference_window=(None if reference_index < 0 else stack.blocks[reference_index].window),
    )


def _companion(payload: Payload, dgp: Any, *, with_reference: bool) -> tuple[Any, int, int]:
    """The companion's three kinds of block, each on its own scramble stream.

    Returns ``(stack, scoring_index, reference_index)``; ``reference_index`` is ``-1`` when no
    ceiling arm runs, since only the ceiling needs one.

    **Three streams and not one, and each separation is load-bearing.**  The *evaluation* blocks
    are what ``P_0 D-hat`` is integrated on.  The *scoring* block is where the component risks
    are read, and it is separate so that the mechanism column and ``root_n_remaining`` are not
    two readings of one randomisation.  The *reference* block is where the ceiling's projection
    is fitted, and it is separate because a reference sharing a scramble with the block the
    remainder is integrated on would make the reference's error and that remainder the same
    random variable, with a covariance nobody can sign.

    Every block is a Sobol block rather than a draw block, so its weights are the law's
    ``g_0(a | W)`` -- ``_check_the_weights_are_the_laws`` refuses the draw-block mistake, whose
    weights are ones, and this is the shape that clears it.
    """
    offset = payload.data_seed % 1_000_003
    evaluation = tuple(QUADRATURE_SEED + offset + i for i in range(QUADRATURE_SCRAMBLES))
    scrambles: tuple[int, ...] = (*evaluation, SCORING_SEED + offset)
    points = [payload.quadrature_points] * len(scrambles)
    if with_reference:
        scrambles = (*scrambles, REFERENCE_SEED + offset)
        points = [*points, payload.reference_points]
        stack = drtmle_remainder.stacked_companion(dgp, points=points, scrambles=scrambles)
        return stack, len(evaluation), len(stack.blocks) - 1
    stack = drtmle_remainder.stacked_companion(dgp, points=points, scrambles=scrambles)
    return stack, len(evaluation), -1


def component_risks(
    fit: Any, *, dgp: Any, mass: np.ndarray, scoring: np.ndarray
) -> dict[str, float]:
    r"""The five held-out component risks of one arm's reduced regressions.

    ``E_0[w (T - m-hat(U))^2]`` on companion rows, exactly
    :func:`~benchmarks.drtmle_reference.held_out_risk`'s quantity and normalisation -- three
    componentwise (:math:`Q_r`, :math:`g_{r,1}`, :math:`g_{r,2}`) and two composite
    (:math:`q_r/g` and :math:`g_{r,2}/g_{r,1}`), which are the two the fit actually divides by.

    **Every companion row is held out by construction**, which is what makes this legitimate
    without a second fit: the companion contributes to no fit, no fold and no score --
    ``tests/unit/test_drtmle_companion.py`` pins that bit for bit -- so an arm's reduced
    regressions, fitted on the sample rows, have seen none of these. The scoring block is
    nevertheless its **own scramble stream**, disjoint from the blocks the remainder is
    integrated on, so that this column and ``root_n_remaining`` are not two readings of one
    randomisation.

    **A risk is not the candidate's own arrays re-scored.**  The predictions come off the
    companion :class:`~cleverly.estimators.reduced.ReducedSet` the fit produced, and the targets
    off :func:`~benchmarks.drtmle_reference.fold_targets`, which builds them from the *law* --
    :math:`Y` never appears in one, because the companion carries :math:`\bar Q_0(a, W)` in its
    outcome column.

    **This ranks and it does not bound**, which is why the roadmap labels the whole group
    *diagnosis and not proof of consistency*: the decomposition leaves a term
    :math:`E_0[w(T-m)^2]` common to every candidate, so a difference of risks is a difference of
    squared errors and **shared** inadequacy is invisible to it.  Two of the five gate, and only
    as a ``worsened`` veto.
    """
    state = fit.nuisance
    companion = getattr(state, "companion", None)
    reduced = None if companion is None else getattr(companion, "reduced", None)
    if companion is None or not reduced:
        return {metric.name: float("nan") for metric in drtmle_reference.METRICS}

    bounds = fit.config.g_bounds
    collected: dict[str, list[float]] = {metric.name: [] for metric in drtmle_reference.METRICS}
    for arm in state.arms:
        truth = drtmle_reference.arm_truth(state, dgp=dgp, arm=float(arm))
        for fold in range(companion.n_folds):
            designs, targets = drtmle_reference.fold_targets(
                state, fold=fold, arm=float(arm), truth=truth, g_bounds=bounds
            )
            del designs  # the arm's own predictions stand in for a fitted reference
            divisors = drtmle_reference.composite_denominators(
                state, fold=fold, arm=float(arm), g_bounds=bounds, reduced=reduced[fold]
            )
            weights = drtmle_reference.metric_weights(mass, divisors)
            block = reduced[fold]
            column = block.column_for(float(arm))
            for metric in drtmle_reference.METRICS:
                name = metric.reduction
                keep = drtmle_reference.fit_mask(name, truth.indicator)
                rows = scoring if keep is None else (scoring & keep)
                predicted = np.asarray(getattr(block, name), dtype=float)[:, column]
                residual = np.asarray(targets[name], dtype=float).reshape(-1) - predicted
                weight = np.asarray(weights[metric.name], dtype=float).reshape(-1)[rows]
                total = float(weight.sum())
                if total <= 0.0 or not np.isfinite(total):
                    continue
                collected[metric.name].append(float(np.dot(residual[rows] ** 2, weight) / total))
    return {
        name: float(np.mean(values)) if values else float("nan")
        for name, values in collected.items()
    }


def _flex_weights(fit: Any, learner: str) -> tuple[float, float]:
    r"""``(mean, min)`` of the flexible candidate's SuperLearner weight.

    **The mean is the one a clause reads and the minimum is a diagnostic beside it**, and that
    ordering is a correction rather than a preference.  The nomination clause originally read
    the *minimum* over 3 reduced regressions x 5 folds -- fifteen values -- and asked it to
    clear ``0.05``.  That is not the question "is this arm the baseline under another name": a
    single fold in which an ensemble happens to put no weight on the flexible candidate is
    ordinary, so the minimum sits at zero for almost any arm.  Measured on the first 41 partial
    draws it was ``0.0000`` at **every quantile including the maximum** for the boosted arm and
    at the median for both spline arms -- so as written the clause could not be passed by any
    arm, and the study would have returned "no nomination" as an artefact of its own predicate.

    That is the failure mode F4's frozen rule had and F5 was written to avoid, so it is repaired
    on the instrument rather than absorbed: ``docs/drtmle/terminal-experiment.md`` records it,
    and it is the F3-closeout precedent -- *the instruments corrected before anything reads
    them* -- and not a threshold moved to clear one.
    """
    if learner in {"glm", "ceiling"}:
        return (float("nan"), float("nan"))
    try:
        diagnostics = fit.extra["drtmle"].diagnostics
    except Exception:  # pragma: no cover - a diagnostic must not fail a fit
        return (float("nan"), float("nan"))
    seen: list[float] = []
    for name in ("qr", "gr1", "gr2"):
        per_regression = diagnostics.get(name)
        if per_regression is None:
            continue
        for per_fold in per_regression:
            # `or ()` is forbidden on either of these: `names` and `weights` may arrive as
            # numpy arrays, and truth-testing one raises "the truth value of an array with more
            # than one element is ambiguous" -- which is how this first surfaced, as a
            # per-arm failure recorded on every draw rather than as an import error.
            raw_names = getattr(per_fold, "names", None)
            raw_weights = getattr(per_fold, "weights", None)
            if raw_names is None or raw_weights is None:
                continue
            names = [str(entry) for entry in raw_names]
            weights = np.asarray(raw_weights, dtype=float).reshape(-1)
            if learner in names and weights.size == len(names):
                seen.append(float(weights[names.index(learner)]))
    if not seen:
        return (float("nan"), float("nan"))
    return (float(np.mean(seen)), float(np.min(seen)))


def _failure_counts(check: Any) -> tuple[int, int]:
    """A failing fit's two causes, counted apart, the way C3c's gate 1 reads them."""
    identity = len(check.identity_failures)
    return identity, len(check.failures) - identity


def _scores(reduction: Any, fluctuation: Any) -> tuple[float, float, float]:
    """The three empirical means at exit, as the loop's own trace last recorded them."""
    if reduction is None or not reduction.trace:
        return (float(fluctuation.relative_score_norm), 0.0, 0.0)
    last = reduction.trace[-1]
    return (float(last[1]), float(last[3]), float(last[2]))


def _bound_witness(fit: Any) -> tuple[bool, float]:
    """Whether the truncation was active at exit, and the initial mechanism's clip share.

    Recorded on every row and **never** used to select or stratify a primary result: selecting a
    comparison on its own realized post-fit label is what F4's row forbids by name.
    """
    try:
        check = fit.validation.score_check(tolerance=VALIDITY_TOLERANCE)
        clipped = max((int(row.clipped) for row in check.rows), default=0)
    except Exception:  # pragma: no cover - a diagnostic must not fail a fit
        clipped = 0
    propensity = np.asarray(fit.nuisance.propensity.values, dtype=float)
    lower, upper = fit.fluctuations["mean"].reduction.bounds
    outside = (propensity < lower) | (propensity > upper)
    share = float(np.mean(outside.any(axis=1))) if outside.ndim > 1 else float(np.mean(outside))
    return bool(clipped > 0), share


def one_draw(payload: Payload) -> list[FitRow]:
    """Every arm on one draw, paired inside the worker.

    Paired inside the worker rather than across passes: the arms see the same rows, the same
    fold seed and the same cell settings, so every difference between two of them is the one
    factor that separates them.  An arm that raises is **recorded** rather than dropped -- an arm
    that raises more often than the baseline is itself a finding, and dropping it would condition
    the comparison on the draws where both happened to work.
    """
    dgp, settings = _law_and_settings(payload.cell, payload.n)
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    truth = dgp.truth()

    lean = _context(payload, dgp, with_reference=False)
    needs_reference = any(ARMS[a].learner == "ceiling" for a in payload.arms)
    full = _context(payload, dgp, with_reference=True) if needs_reference else None

    rows: list[FitRow] = []
    for arm in payload.arms:
        spec = ARMS[arm]
        ceiling = spec.learner == "ceiling"
        context = full if ceiling and full is not None else lean
        stack = context.stack
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = arm_estimator(
                    arm,
                    settings,
                    random_state=payload.fold_seed,
                    evaluation=stack.frame,
                    dgp=dgp if ceiling else None,
                    window=context.reference_window if ceiling else None,
                    row_weights=stack.weights if ceiling else None,
                )
                result = estimator.fit(frame, outcome="Y", treatment="A")
                fit = result.single()
                remainder = drtmle_remainder.remainder_rows(
                    fit,
                    dgp,
                    n=payload.n,
                    bounds=fit.config.g_bounds,
                    row_weights=stack.weights,
                    windows=context.windows,
                    truths=context.truths,
                )
                fluctuation = fit.fluctuations["mean"]
                reduction = fluctuation.reduction
                check = fit.validation.score_check(tolerance=VALIDITY_TOLERANCE)
                identity_failures, score_failures = _failure_counts(check)
                score_8, score_9, score_10 = _scores(reduction, fluctuation)
                active, share = _bound_witness(fit)
                flex_mean, flex_min = _flex_weights(fit, spec.learner)
                risks = component_risks(fit, dgp=dgp, mass=stack.weights, scoring=context.scoring)
        except Exception as exc:  # recorded and reported, never swallowed
            rows.extend(_failed(payload, arm, f"{type(exc).__name__}: {exc}", truth))
            continue
        seconds = time.perf_counter() - started
        by_estimand = {row.estimand: row for row in remainder}
        impl = resolved_implementations().get(arm, {}).get("regression", spec.learner)
        for name, estimate in fit.estimates.items():
            row = by_estimand.get(name)
            rows.append(
                FitRow(
                    cohort=payload.cohort,
                    cell=payload.cell,
                    n=payload.n,
                    data_seed=payload.data_seed,
                    fold_seed=payload.fold_seed,
                    arm=arm,
                    estimand=name,
                    psi=float(estimate.psi),
                    truth=float(truth.get(name, float("nan"))),
                    abs_error=abs(float(estimate.psi) - float(truth.get(name, float("nan")))),
                    std_error=float(estimate.std_error),
                    root_n_remaining=float("nan") if row is None else float(row.root_n_remaining),
                    score_8=score_8,
                    score_9=score_9,
                    score_10=score_10,
                    identity_failures=identity_failures,
                    score_failures=score_failures,
                    valid=bool(check.passed),
                    rounds=0 if reduction is None else int(reduction.rounds),
                    closing=0 if reduction is None else int(reduction.closing),
                    exit_reason="" if reduction is None else str(reduction.exit_reason),
                    failure=""
                    if reduction is None or reduction.failure is None
                    else str(reduction.failure),
                    bound_active=active,
                    initial_clip_share=share,
                    flex_weight_mean=flex_mean,
                    flex_weight_min=flex_min,
                    risk_qr=risks.get("qr", float("nan")),
                    risk_gr1=risks.get("gr1", float("nan")),
                    risk_gr2=risks.get("gr2", float("nan")),
                    risk_h3=risks.get("h3", float("nan")),
                    risk_h2=risks.get("h2", float("nan")),
                    impl=impl,
                    seconds=seconds,
                )
            )
    return rows


def _failed(payload: Payload, arm: str, error: str, truth: Mapping[str, float]) -> list[FitRow]:
    """A draw an arm raised on, recorded as invalid rather than dropped."""
    return [
        FitRow(
            cohort=payload.cohort,
            cell=payload.cell,
            n=payload.n,
            data_seed=payload.data_seed,
            fold_seed=payload.fold_seed,
            arm=arm,
            estimand=name,
            psi=float("nan"),
            truth=float(truth.get(name, float("nan"))),
            abs_error=float("nan"),
            std_error=float("nan"),
            root_n_remaining=float("nan"),
            score_8=float("nan"),
            score_9=float("nan"),
            score_10=float("nan"),
            identity_failures=0,
            score_failures=0,
            valid=False,
            rounds=0,
            closing=0,
            exit_reason="",
            failure="",
            bound_active=False,
            initial_clip_share=float("nan"),
            flex_weight_mean=float("nan"),
            flex_weight_min=float("nan"),
            risk_qr=float("nan"),
            risk_gr1=float("nan"),
            risk_gr2=float("nan"),
            risk_h3=float("nan"),
            risk_h2=float("nan"),
            impl="",
            seconds=float("nan"),
            error=error,
        )
        for name in ESTIMANDS
    ]


# ------------------------------------------------------------------ the ceiling's two readings


def ceiling_crossfit_reading(
    *, n: int = 600, cell: str = "q-drift", seed: int = 20260205, points: int = 2_048
) -> list[dict[str, Any]]:
    r"""Are the ceiling's two cross-fitting cells the same object?  Exactly, not over a cohort.

    The roadmap's arm matrix crosses the ceiling with ``reduced_crossfit``, which would be an
    eighth arm.  It is a **null by construction** and this is the reading that says so rather
    than a cohort that would spend a seventh of the budget measuring a zero:

    * ``reduced_crossfit`` acts only inside ``fit_reduced``, through ``NuisanceEstimates.inner``,
      on the reduced regressions' **training** rows
      (``src/cleverly/estimators/reduced.py``'s ``inner = nuisance.inner if crossfit == "nested"``
      and ``_roles``' ``inner.propensity[fold]`` / ``inner.outcome[fold]``);
    * :class:`~benchmarks.drtmle_reference.ReferenceReductionDRTMLE` replaces the produced set in
      ``_nuisances`` and replaces ``spec.refit`` in ``_reduction``, so **no array**
      ``fit_reduced`` produces is ever read.

    **Not on the committed trace fixture**, and the reason is structural rather than a
    preference: a reference is fitted against the *law's own* conditional means through
    :func:`~benchmarks.drtmle_reference.arm_truth`, and the trace fixture is 200 frozen rows with
    closed-form nuisances injected through sklearn-shaped learners -- it has no ``DGP`` behind it
    to be the truth of.  So this reads on one deterministic tier-2 draw at a fixed seed instead.
    It is still an identity and not an estimate: one draw, two keywords, arrays compared
    element by element, no sampling anywhere in the comparison.

    If this ever stops holding, the eighth arm comes back -- which is why
    ``tests/unit/test_drtmle_f5.py`` asserts it rather than this docstring merely claiming it.
    """
    if points < MINIMUM_REFERENCE_POINTS:
        raise ValueError(
            f"a reference block of {points} points offers {points} rows to Q_r, which is masked "
            f"to one arm; spline({CEILING_KNOTS}) needs {MINIMUM_REFERENCE_POINTS}"
        )
    dgp = drtmle_tier2.base_law()
    frame, _ = dgp.sample(n, seed=seed)
    settings = dict(drtmle_tier2.settings(cell, n))
    stack = drtmle_remainder.stacked_companion(
        dgp, points=points, scrambles=(REFERENCE_SEED + seed % 1_000_003,)
    )
    produced: dict[str, Any] = {}
    for crossfit in ("pooled", "nested"):
        estimator = drtmle_reference.ReferenceReductionDRTMLE(
            dgp=dgp,
            reference=drtmle_reference.SplineProjection(CEILING_KNOTS),
            window=stack.blocks[0].window,
            row_weights=stack.weights,
            reduced_crossfit=crossfit,
            evaluation=stack.frame,
            random_state=seed,
            **settings,
            **reduced_library("glm", seed),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = estimator.fit(frame, outcome="Y", treatment="A").single()
        produced[crossfit] = {
            "initial": fit.nuisance.reduced,
            "converged": fit.fluctuations["mean"].reduction.reduced,
        }
    rows: list[dict[str, Any]] = []
    for stage in ("initial", "converged"):
        for quantity in ("qr", "gr1", "gr2"):
            left = np.asarray(getattr(produced["pooled"][stage], quantity), dtype=float)
            right = np.asarray(getattr(produced["nested"][stage], quantity), dtype=float)
            rows.append(
                {
                    "stage": stage,
                    "quantity": quantity,
                    "identical": bool(np.array_equal(left, right)),
                    "worst": float(np.max(np.abs(left - right))),
                }
            )
    return rows


# ------------------------------------------------------------------ the contrasts


def _paired_interval(values: np.ndarray, seed: int) -> tuple[float, float, float, float]:
    """A paired difference's mean, its percentile interval and its spread."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(BOOTSTRAP, finite.size), replace=True).mean(axis=1)
    lower, upper = np.percentile(draws, INTERVAL)
    return (float(finite.mean()), float(lower), float(upper), float(finite.std(ddof=1)))


def _cohort_statistic(column: str, rows: Sequence[FitRow]) -> float:
    """A set functional of one arm's draws in one cell: the ``statistic="cohort"`` columns."""
    good = [row for row in rows if not row.error]
    if not good:
        return float("nan")
    psi = np.asarray([row.psi for row in good], dtype=float)
    truth = np.asarray([row.truth for row in good], dtype=float)
    se = np.asarray([row.std_error for row in good], dtype=float)
    n = float(good[0].n)
    finite = np.isfinite(psi) & np.isfinite(truth)
    if not finite.any():
        return float("nan")
    error = psi[finite] - truth[finite]
    if column == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if column == "root_n_bias":
        return float(math.sqrt(n) * np.mean(error))
    if column == "empirical_sd":
        return float(np.std(psi[finite], ddof=1)) if finite.sum() > 1 else float("nan")
    if column == "se_ratio":
        spread = np.std(psi[finite], ddof=1) if finite.sum() > 1 else float("nan")
        mean_se = np.nanmean(se[finite])
        return float(mean_se / spread) if spread and math.isfinite(spread) else float("nan")
    if column == "abs_coverage_gap":
        half = 1.959963984540054 * se[finite]
        covered = np.abs(error) <= half
        return float(abs(np.mean(covered) - 0.95))
    if column == "bound_active":
        return float(np.mean([row.bound_active for row in good]))
    if column == "seconds":
        return float(np.nanmean([row.seconds for row in good]))
    return float("nan")


def contrast_rows(rows: Sequence[FitRow], *, seed: int = COHORT_SEED) -> list[ContrastRow]:
    """Every arm against the baseline, on every declared column, in every cell.

    Raises on a pilot row.  The pilot is timing only, and a harness that merely *documented*
    that would be one edit away from reading it.
    """
    if any(row.cohort == SIZING_COHORT for row in rows):
        raise ValueError(
            f"{SIZING_COHORT!r} rows reached contrast_rows: the pilot measures cost and sizes "
            "nothing -- sizing comes from F4's committed PILOT_PAIRED_SPREAD and nothing else"
        )
    primary = [row for row in rows if row.estimand == PRIMARY_ESTIMAND]
    indexed: dict[tuple[Any, ...], FitRow] = {
        (r.cohort, r.cell, r.n, r.data_seed, r.arm): r for r in primary
    }
    groups = sorted({(r.cohort, r.cell, r.n) for r in primary})
    out: list[ContrastRow] = []
    for cohort, cell, n in groups:
        here = [r for r in primary if (r.cohort, r.cell, r.n) == (cohort, cell, n)]
        seeds = sorted({r.data_seed for r in here})
        base_rows = [r for r in here if r.arm == BASELINE_ARM]
        for arm, spec in ARMS.items():
            if arm == BASELINE_ARM:
                continue
            arm_rows = [r for r in here if r.arm == arm]
            if not arm_rows:
                continue
            for column in COLUMNS:
                if column.statistic == "cohort":
                    left = _cohort_statistic(column.name, arm_rows)
                    right = _cohort_statistic(column.name, base_rows)
                    difference = left - right
                    mean, lower, upper, sd = _bootstrap_cohort(
                        column.name, arm_rows, base_rows, seed
                    )
                    baseline_level = right
                else:
                    paired = []
                    for data_seed in seeds:
                        a = indexed.get((cohort, cell, n, data_seed, arm))
                        b = indexed.get((cohort, cell, n, data_seed, BASELINE_ARM))
                        if a is None or b is None or a.error or b.error:
                            continue
                        paired.append(
                            float(getattr(a, column.name)) - float(getattr(b, column.name))
                        )
                    mean, lower, upper, sd = _paired_interval(np.asarray(paired, dtype=float), seed)
                    baseline_level = float(
                        np.nanmean([getattr(r, column.name) for r in base_rows] or [np.nan])
                    )
                    difference = mean
                band = band_for(column, cell, baseline_level)
                out.append(
                    ContrastRow(
                        cohort=cohort,
                        cell=cell,
                        n=n,
                        arm=arm,
                        role=spec.role,
                        column=column.name,
                        group=column.group,
                        draws=len([r for r in arm_rows if not r.error]),
                        mean=difference,
                        lower=lower,
                        upper=upper,
                        paired_sd=sd,
                        band=band,
                        verdict=verdict(lower, upper, band, column.orientation),
                    )
                )
    return out


def _bootstrap_cohort(
    column: str, arm_rows: Sequence[FitRow], base_rows: Sequence[FitRow], seed: int
) -> tuple[float, float, float, float]:
    """A set functional's difference, bootstrapped over **draw indices** with both arms paired.

    One resample of the draw indices per replicate, both arms recomputed on that same resample,
    so the interval is of a paired difference rather than of two independent estimates whose
    difference happens to be taken.
    """
    by_seed_arm = {r.data_seed: r for r in arm_rows if not r.error}
    by_seed_base = {r.data_seed: r for r in base_rows if not r.error}
    shared = sorted(set(by_seed_arm) & set(by_seed_base))
    if len(shared) < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    point = _cohort_statistic(column, [by_seed_arm[s] for s in shared]) - _cohort_statistic(
        column, [by_seed_base[s] for s in shared]
    )
    draws = np.empty(BOOTSTRAP, dtype=float)
    index = np.arange(len(shared))
    for b in range(BOOTSTRAP):
        picked = rng.choice(index, size=index.size, replace=True)
        chosen = [shared[i] for i in picked]
        draws[b] = _cohort_statistic(column, [by_seed_arm[s] for s in chosen]) - _cohort_statistic(
            column, [by_seed_base[s] for s in chosen]
        )
    finite = draws[np.isfinite(draws)]
    if finite.size < 2:
        return (point, float("nan"), float("nan"), float("nan"))
    lower, upper = np.percentile(finite, INTERVAL)
    return (point, float(lower), float(upper), float(finite.std(ddof=1)))


# ------------------------------------------------------------------ the nomination


def refuse_dead_gates(rows: Sequence[ContrastRow]) -> list[str]:
    """Every gating column that produced no finite reading anywhere.  Empty is the only pass.

    **A veto that cannot fire is not a veto**, and it reads in a table exactly like one that
    fired and found nothing.  F4's second rule defect was a band that made a verdict
    unreachable; this is the same failure one layer down -- a column whose *values* are absent,
    so every interval is ``nan`` and every verdict is ``unresolved``, which fires no clause.

    Checked before a nomination is taken rather than after, so an unimplemented or silently
    broken column stops the study instead of quietly widening what qualifies.
    """
    complaints: list[str] = []
    for column in COLUMNS:
        if not column.gates:
            continue
        mine = [r for r in rows if r.column == column.name]
        if not mine:
            complaints.append(f"{column.name}: gating column produced no rows at all")
            continue
        if not any(math.isfinite(r.mean) for r in mine):
            complaints.append(
                f"{column.name}: gating column has no finite reading in any cell -- a veto that "
                "cannot fire reads exactly like one that fired and found nothing"
            )
    return complaints


def nominate(rows: Sequence[ContrastRow], fits: Sequence[FitRow]) -> dict[str, Any]:
    """At most one feasible configuration, under the rule frozen before the first fit.

    The clauses, and they are a conjunction rather than a score:

    1. the arm is **nominable** -- a ceiling is not a procedure a caller can run, and
       ``boost-pooled`` is diagnostic because A1b's pooled premise is not closed for it;
    2. ``root_n_remaining`` is ``improved`` or ``equivalent`` in **every** (cell, size), and
       ``improved`` in at least one;
    3. no gating column ever reads ``worsened``;
    4. **zero** identity failures on every one of the arm's fits -- an exact clause, because
       C3c recorded zero across 6,000 fits and a banded version would turn the one certainty
       into a statistic;
    5. the flexible candidate carries real weight -- its **mean** SuperLearner weight clears
       :data:`FLEX_WEIGHT_FLOOR` in :data:`FLEX_WEIGHT_SHARE` of fits, since an arm that
       collapsed onto ``mean`` is the baseline under another name.  The mean and not the
       minimum: see :data:`FLEX_WEIGHT_FLOOR` for the correction and why it was needed;
    6. every clause above holds on the **audit** cohort as well as the selection cohort.

    Ties are broken by declared arm order, simplest first.
    """
    if any(row.cohort == SIZING_COHORT for row in rows):
        raise ValueError(f"{SIZING_COHORT!r} rows reached nominate: the pilot decides nothing")
    dead = refuse_dead_gates(rows)
    if dead:
        raise ValueError(
            "a nomination may not be taken while a gating column is dead: " + "; ".join(dead)
        )

    reasons: dict[str, list[str]] = {}
    eligible: list[str] = []
    cohorts = sorted({row.cohort for row in rows})
    for arm, spec in ARMS.items():
        if arm == BASELINE_ARM:
            continue
        why: list[str] = []
        if not spec.nominable:
            why.append(f"not nominable: {spec.why}")
            reasons[arm] = why
            continue
        mine = [r for r in rows if r.arm == arm]
        remaining = [r for r in mine if r.column == "root_n_remaining"]
        if not remaining:
            why.append("no root_n_remaining rows")
        for cohort in cohorts:
            here = [r for r in remaining if r.cohort == cohort]
            if not here:
                why.append(f"{cohort}: no rows -- an effect must reproduce on the audit cohort")
                continue
            bad = [r for r in here if r.verdict not in {"improved", "equivalent"}]
            if bad:
                why.append(
                    f"{cohort}: root_n_remaining reads "
                    + ", ".join(sorted({f"{r.verdict} at {r.cell} n={r.n}" for r in bad}))
                )
            if not any(r.verdict == "improved" for r in here):
                why.append(f"{cohort}: root_n_remaining never improved")
        worsened = [r for r in mine if COLUMN_BY_NAME[r.column].gates and r.verdict == "worsened"]
        if worsened:
            why.append(
                "worsened on a gating column: "
                + ", ".join(sorted({f"{r.column} at {r.cell} n={r.n}" for r in worsened}))
            )
        arm_fits = [f for f in fits if f.arm == arm and not f.error]
        identity = sum(f.identity_failures for f in arm_fits)
        if identity:
            why.append(f"{identity} identity failure(s) -- C3c recorded zero across 6,000 fits")
        weights = np.asarray(
            [f.flex_weight_mean for f in arm_fits if math.isfinite(f.flex_weight_mean)],
            dtype=float,
        )
        if weights.size:
            share = float(np.mean(weights >= FLEX_WEIGHT_FLOOR))
            if share < FLEX_WEIGHT_SHARE:
                why.append(
                    f"the flexible candidate's mean SuperLearner weight was below "
                    f"{FLEX_WEIGHT_FLOOR:g} in {1 - share:.0%} of fits -- this arm is the "
                    "baseline under another name"
                )
        if why:
            reasons[arm] = why
        else:
            eligible.append(arm)

    order = [name for name in ARMS if name in eligible]
    return {
        "nominated": order[0] if order else "none",
        "eligible": order,
        "rejected": reasons,
        "rule": "root_n_remaining primary; gating columns as vetoes; reproduced on the audit "
        "cohort; ties broken by declared arm order",
    }


# ------------------------------------------------------------------ the frozen manifest


def _environment() -> dict[str, Any]:
    """What the study was built and dispatched on, recorded so a rerun can be compared to it."""
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        ).stdout
    except Exception:  # pragma: no cover - the digest is a record, not a gate
        freeze = ""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest() if freeze else "",
        "resolved": resolved_implementations(),
        "thread_limit": 1,
    }


def prereg(
    *,
    seed: int = COHORT_SEED,
    draws: int | None = None,
    replicates: int = CONFIRM_REPLICATES,
) -> dict[str, Any]:
    """The whole frozen design, **both phases**, as a dictionary a commit can carry.

    One manifest rather than two, because F5's row requires the confirmation's seeds, replicate
    counts, margins, exclusions and decision rules to be committed *before phase 1 begins* -- so
    that the only thing crossing between the phases is which candidate was nominated.
    """
    count = max(SIZE_DRAWS.values()) if draws is None else int(draws)
    return {
        "environment": _environment(),
        "rule": frozen_rule(),
        "phase1": {
            "arms": list(ARMS),
            "arms_dropped": dict(ARMS_DROPPED),
            "cells": list(CELLS),
            "cohorts": {
                name: [list(pair) for pair in pairs]
                for name, pairs in cohort_seeds(seed, count).items()
            },
            "draws": count,
            "draws_by_size": {str(k): v for k, v in sorted(SIZE_DRAWS.items())},
            "powered": {
                "600": "a material move, not equivalence -- F4's declared limit at this size",
                "2400": "both",
            },
            "quadrature_points": QUADRATURE_POINTS,
            "quadrature_scrambles": QUADRATURE_SCRAMBLES,
            "reference_points": REFERENCE_POINTS,
            "seed": seed,
            "sizes": list(SIZES),
            "sizing_stream": SIZING_STREAM,
            "tier": TIER,
        },
        "phase2": {
            "arms": ["baseline", "nomination", "ceiling", "tmle"],
            "batches": {
                "A": {"seed": CONFIRM_SEED_A, "replicates": replicates},
                "B": {"seed": CONFIRM_SEED_B, "replicates": replicates},
            },
            "cells": [*CELLS, STRESS_CELL],
            "clause_3": {
                "status": "not feasible",
                "why": "the published Biometrika paper is recorded 'Not read here' in "
                "docs/references.md, 'in repository: no' in docs/drtmle/theorem-concordance.md, "
                "and docs/drtmle/investigation-log.md records that only the working paper's "
                "Theorem 1 and appendices A-C were obtainable. Its simulation section is in "
                "neither, so a reproduction would be a DGP attributed to a paper nobody read",
                "instead": "an applied stress cell (benchmarks/drtmle_stress.py) discharging "
                "gate 2 clause 4, which C3c left unread",
            },
            "exclusions": [
                "the ceiling arm does not run at n=1,200: that size exists for clause 4's "
                "trend, which is read on the baseline and the nomination",
                "the ceiling arm does not run in the stress cell: no branch reads one there",
                "the stress cell's remainder is NOT read against clause 4's vanishing trend -- "
                "with both primaries inconsistent no theorem predicts one, and item 13 closes "
                "on the drift cells only",
                "the stress cell's coverage is descriptive and is not a release number",
            ],
            "nominal": 0.95,
            "sizes": list(CONFIRM_SIZES),
            "stress_sizes": list(drtmle_stress.SIZES),
            "monte_carlo_rule": "compatible iff |phat - 0.95| <= 1.96 * sqrt(phat (1 - phat) / M)"
            f", at M = {replicates} per batch (MC se 0.0097, band +/- 0.019)",
        },
        "disjointness": {
            "f4_seed": drtmle_construction.COHORT_SEED,
            "f4_draws": max(drtmle_construction.SIZE_DRAWS.values()),
            "checked": "phase-1 cohorts against each other and against F4's; confirmation "
            "batches against each other and against both phase-1 cohorts",
        },
    }


def write_prereg(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write the manifest sorted and indented, so a commit of it is reviewable line by line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_prereg(
    manifest: Mapping[str, Any],
    *,
    phase: str,
    cohort: str = "selection",
    seeds: Sequence[tuple[int, int]] = (),
    draws_declared: int | None = None,
) -> list[str]:
    """Every reason this run may not be read against that manifest.  Empty is the only pass."""
    complaints: list[str] = []

    rule = manifest.get("rule") or {}
    if not rule:
        complaints.append("rule: the manifest carries no rule block")
    else:
        live = frozen_rule()
        for key, value in live.items():
            if rule.get(key) != value:
                complaints.append(
                    f"rule: {key} moved after the freeze -- a verdict read against a bar chosen "
                    "to clear is not a verdict"
                )
        for entry in rule.get("columns", ()):
            band = entry.get("band")
            if not isinstance(band, (int, float)) or not math.isfinite(band) or band <= 0:
                complaints.append(
                    f"rule: column {entry.get('name')!r} has band {band!r}; a non-finite or "
                    "non-positive band makes the third verdict unreachable, which is F4's "
                    "second rule defect"
                )

    environment = manifest.get("environment") or {}
    if environment.get("resolved") != resolved_implementations():
        complaints.append(
            "environment: the resolved learner implementations differ from this box's -- two "
            "function classes under one arm name make the contract's entropy row ambiguous"
        )
    complaints.extend(refuse_on_fallback())

    if phase == "select" and not manifest.get("phase2"):
        complaints.append(
            "phase2: absent from the manifest -- the confirmation's seeds, replicate counts, "
            "margins, exclusions and decision rules are committed BEFORE phase 1 begins"
        )

    phase1 = manifest.get("phase1") or {}
    declared = {
        name: {int(pair[0]) for pair in pairs}
        for name, pairs in (phase1.get("cohorts") or {}).items()
    }
    if seeds:
        if cohort not in COHORTS:
            complaints.append(f"cohorts: {cohort!r} is not one of {list(COHORTS)}")
            return complaints
        mine = {int(data) for data, _ in seeds}
        for other in COHORTS:
            if other == cohort:
                continue
            overlap = sorted(mine & declared.get(other, set()))
            if overlap:
                complaints.append(
                    f"cohorts: {len(overlap)} draw(s) of the {cohort} run are {other} draws -- "
                    f"an effect would be reproduced on the draws that produced it, first at "
                    f"data seed {overlap[0]}"
                )
        unknown = sorted(mine - declared.get(cohort, set()))
        if unknown:
            complaints.append(
                f"cohorts: {len(unknown)} draw(s) are not in the committed {cohort} cohort, "
                f"first at data seed {unknown[0]}"
            )
        expected = len(declared.get(cohort, set())) if draws_declared is None else draws_declared
        if expected and len(mine) < math.ceil(COMPLETENESS_FRACTION * expected):
            complaints.append(
                f"cohorts: {len(mine)} of {expected} declared {cohort} draws, below the "
                f"{COMPLETENESS_FRACTION:g} completeness clause -- the cell is unresolved by rule"
            )

    # F4's draws may not be reused for either inferential phase.
    f4 = drtmle_construction.cohort_seeds(
        drtmle_construction.COHORT_SEED, max(drtmle_construction.SIZE_DRAWS.values())
    )
    f4_data = {int(a) for pairs in f4.values() for a, _ in pairs}
    for name, ours in declared.items():
        shared = sorted(ours & f4_data)
        if shared:
            complaints.append(
                f"cohorts: {len(shared)} of the {name} cohort's draws are F4's, first at data "
                f"seed {shared[0]} -- both F4 cohorts have been read"
            )

    phase2 = manifest.get("phase2") or {}
    batches = phase2.get("batches") or {}
    batch_seeds = {
        name: {int(a) for a, _ in confirm_seeds(int(block["seed"]), int(block["replicates"]))}
        for name, block in batches.items()
    }
    names = sorted(batch_seeds)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = batch_seeds[left] & batch_seeds[right]
            if shared:
                complaints.append(
                    f"phase2: batches {left} and {right} share {len(shared)} draw(s) -- the "
                    "independent second batch is what clause 8 is"
                )
        for name, ours in declared.items():
            shared = batch_seeds[left] & ours
            if shared:
                complaints.append(
                    f"phase2: batch {left} shares {len(shared)} draw(s) with the phase-1 "
                    f"{name} cohort -- both phases run on fresh data seeds"
                )
    return complaints


# ------------------------------------------------------------------ the report


FIT_HEADERS = (
    "cell",
    "n",
    "arm",
    "draws",
    "sqrt(n) R",
    "score fails",
    "identity",
    "rounds",
    "flex wt",
    "s/fit",
)
CONTRAST_HEADERS = (
    "cell",
    "n",
    "arm",
    "column",
    "draws",
    "mean",
    "95% interval",
    "band",
    "verdict",
)


def _fit_table(rows: Sequence[FitRow]) -> str:
    primary = [r for r in rows if r.estimand == PRIMARY_ESTIMAND]
    body = []
    for cell, n, arm in sorted({(r.cell, r.n, r.arm) for r in primary}):
        group = [r for r in primary if (r.cell, r.n, r.arm) == (cell, n, arm)]
        good = [r for r in group if not r.error]
        remaining = np.asarray([r.root_n_remaining for r in good], dtype=float)
        remaining = remaining[np.isfinite(remaining)]
        flex = np.asarray([r.flex_weight_mean for r in good], dtype=float)
        flex = flex[np.isfinite(flex)]
        body.append(
            (
                cell,
                str(n),
                arm,
                f"{len(good)}/{len(group)}",
                "--" if remaining.size == 0 else f"{remaining.mean():.4f}",
                str(sum(r.score_failures for r in good)),
                str(sum(r.identity_failures for r in good)),
                "--" if not good else f"{np.mean([r.rounds for r in good]):.1f}",
                "--" if flex.size == 0 else f"{flex.mean():.3f}",
                "--" if not good else f"{np.nanmean([r.seconds for r in good]):.1f}",
            )
        )
    return format_table(FIT_HEADERS, body)


def _contrast_table(rows: Sequence[ContrastRow], columns: Sequence[str]) -> str:
    body = [
        (
            row.cell,
            str(row.n),
            row.arm,
            row.column,
            str(row.draws),
            f"{row.mean:+.4f}",
            f"[{row.lower:+.4f}, {row.upper:+.4f}]",
            f"{row.band:.4f}",
            row.verdict,
        )
        for row in rows
        if row.column in set(columns)
    ]
    return format_table(CONTRAST_HEADERS, body)


PILOT_HEADERS = ("cell", "n", "arm", "fits", "median s", "mean s", "worst s")


def _pilot_table(rows: Sequence[FitRow]) -> str:
    primary = [r for r in rows if r.estimand == PRIMARY_ESTIMAND]
    body = []
    for cell, n, arm in sorted({(r.cell, r.n, r.arm) for r in primary}):
        group = [
            r
            for r in primary
            if (r.cell, r.n, r.arm) == (cell, n, arm) and not r.error and math.isfinite(r.seconds)
        ]
        if not group:
            body.append((cell, str(n), arm, "0", "--", "--", "--"))
            continue
        seconds = np.asarray([r.seconds for r in group], dtype=float)
        body.append(
            (
                cell,
                str(n),
                arm,
                str(len(group)),
                f"{np.median(seconds):.1f}",
                f"{seconds.mean():.1f}",
                f"{seconds.max():.1f}",
            )
        )
    return format_table(PILOT_HEADERS, body)


# ------------------------------------------------------------------ the CLI

PHASES = (
    "pilot",
    "prereg",
    "select",
    "nominate",
    "ceiling-crossfit",
    "cost",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--prereg", type=Path, default=Path("evidence/f5-terminal/prereg.json"))
    parser.add_argument("--cohort", choices=COHORTS, default="selection")
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--draws", type=int, default=None, help="debug only outside --phase pilot")
    parser.add_argument("--quadrature-points", type=int, default=QUADRATURE_POINTS)
    parser.add_argument("--reference-points", type=int, default=REFERENCE_POINTS)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="workers; 0 takes default_jobs(), which is the LOGICAL count less two because "
        "every fit here is single-threaded and a worker occupies one hardware thread",
    )
    parser.add_argument("--seed", type=int, default=COHORT_SEED)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/drtmle-f5"))
    return parser


def _emit(payload: str, label: str) -> None:
    """Print a result so it can be recovered from a job log, digest first."""
    raw = payload.encode("utf-8")
    print(f"{label}-SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"{label}-BYTES {len(raw)}")
    print(f"--- BEGIN {label}.gz.base64 ---")
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    for start in range(0, len(encoded), 76):
        print(encoded[start : start + 76])
    print(f"--- END {label}.gz.base64 ---")


def _write_rows(rows: Sequence[Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def default_jobs() -> int:
    r"""How many workers to run, when ``--jobs`` is not given.

    **Every fit here is single-threaded on purpose** -- ``cleverly.learners.set_thread_limit``
    holds the nuisance fits to one thread and this module pins LightGBM's ``n_jobs`` -- so
    parallelism is across draws and a worker occupies exactly one hardware thread.  Measured on
    the phase-1 cohort at ``--jobs 9``: 8.89 cores busy, two OS threads per worker, 98.8%
    occupancy each.  Nothing was stalled or contended; there were simply more hardware threads
    than workers.

    So the default is the **logical** count less a small reserve, not the physical count.  On a
    hybrid part -- the box this was measured on is 6 P-cores with SMT plus 4 E-cores, so 10
    physical and 16 logical -- ``--jobs 9`` left all six SMT siblings idle and read as 56% in a
    task manager, which counts against logical processors.

    **The gain from the SMT siblings is real and is not a doubling**: two threads on one
    physical core share its execution units, so the honest expectation is roughly a quarter
    more throughput rather than twice.  The reserve keeps the parent's bootstrap, JSON and
    flush work off the critical path.
    """
    logical = os.cpu_count() or 2
    return max(1, logical - 2)


def _expected_cost(payload: Payload) -> float:
    """A proxy for how long one draw will take, for longest-first dispatch.

    The sample size, because every payload carries the same arm set and every arm's cost rises
    with ``n``.  A proxy rather than the pilot's measured table: a schedule that reads committed
    timings would go stale the moment an arm moved, and the ordering only has to be roughly
    right to help.
    """
    return float(payload.n)


def _run(payloads: Sequence[Payload], jobs: int, out: Path, stamp: str) -> list[FitRow]:
    """Dispatch longest-first, flushing each completed draw immediately.

    Two things about this, and both were measured rather than assumed.

    **The flush is per draw.**  F4 wrote once at the end, which is fine at two hours and not at
    ten: a crash at hour eight has to cost one draw rather than the cohort.

    **The order is longest-first**, which is the classic LPT heuristic and is worth a few
    percent of makespan here.  Built in cell-major, size-minor order the payloads put every
    ``n = 600`` draw before every ``n = 2,400`` one, so the *long* tasks were dispatched last
    and the run ended with a ragged tail of them while workers idled.  Sorting by ``n``
    descending puts the long tasks in first, where there is still work to pack around them.
    ``joblib`` dispatches dynamically at ``batch_size=1``, so the order is a scheduling hint and
    changes no result: :func:`one_draw` is a pure function of its payload, and the artefact is
    keyed by ``(cohort, cell, n, data_seed, arm)`` rather than by position.
    """
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp}.jsonl"
    rows: list[FitRow] = []
    ordered = sorted(payloads, key=lambda p: (-_expected_cost(p), p.cell, p.data_seed))
    produced = Parallel(n_jobs=jobs, batch_size=1, return_as="generator_unordered")(
        delayed(one_draw)(p) for p in ordered
    )
    done = 0
    with path.open("w", encoding="utf-8") as handle:
        for group in produced:
            for row in group:
                handle.write(json.dumps(asdict(row)) + "\n")
                rows.append(row)
            handle.flush()
            done += 1
            print(f"  {done}/{len(ordered)} draws", flush=True)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    if args.jobs <= 0:
        args.jobs = default_jobs()

    complaints = refuse_on_fallback()
    if complaints:
        for line in complaints:
            print(f"error: {line}", file=sys.stderr)
        return 1

    if args.phase == "prereg":
        manifest = prereg(seed=args.seed, draws=args.draws)
        path = write_prereg(manifest, args.out / "prereg.json")
        print(f"wrote {path}")
        print()
        print(format_rule_table())
        print()
        print(
            "cohorts: "
            + ", ".join(f"{k}={len(v)}" for k, v in manifest["phase1"]["cohorts"].items())
            + f"; confirmation {CONFIRM_REPLICATES} per batch x 2 batches"
        )
        _emit(path.read_text(encoding="utf-8"), "PREREG")
        return 0

    if args.phase == "ceiling-crossfit":
        rows = ceiling_crossfit_reading()
        print(
            format_table(
                ("stage", "quantity", "identical", "worst |diff|"),
                [
                    (
                        r["stage"],
                        r["quantity"],
                        "yes" if r["identical"] else "NO",
                        f"{r['worst']:.3e}",
                    )
                    for r in rows
                ],
            )
        )
        print()
        print(
            "Every row identical is what makes the ceiling's nested cell a null by construction\n"
            "and the eighth arm unnecessary. A single 'NO' brings that arm back."
        )
        _emit("".join(json.dumps(r) + "\n" for r in rows), "CEILING-CROSSFIT")
        return 0 if all(r["identical"] for r in rows) else 1

    if args.phase == "pilot":
        draws = args.draws if args.draws is not None else 2
        seeds = sizing_seeds(args.seed, draws)
        payloads = [
            Payload(
                cohort=SIZING_COHORT,
                cell=cell,
                n=n,
                data_seed=int(d),
                fold_seed=int(f),
                arms=tuple(a for a in ARMS if a in set(args.arms)),
                quadrature_points=args.quadrature_points,
                reference_points=args.reference_points,
            )
            for cell in args.cells
            for n in args.sizes
            for d, f in seeds
        ]
        fits = sum(len(p.arms) for p in payloads)
        print(
            f"F5 timing pilot: {len(payloads)} draws, {fits} fits, jobs={args.jobs}\n"
            f"  arms: {', '.join(payloads[0].arms)}\n"
            "  This measures cost and NOTHING else. Sizing comes from F4's committed\n"
            "  PILOT_PAIRED_SPREAD; contrast_rows() and nominate() raise on these rows."
        )
        stamp = time.strftime("%Y%m%dT%H%M%S")
        started = time.perf_counter()
        rows = _run(payloads, args.jobs, args.out, f"{stamp}-pilot")
        wall = time.perf_counter() - started
        print()
        print(_pilot_table(rows))
        print()
        print(f"wall clock {wall / 60:.1f} min at --jobs {args.jobs}")
        errored = sorted({f"{r.arm}: {r.error}" for r in rows if r.error})
        if errored:
            print()
            print("errors:")
            for line in errored:
                print(f"  {line}")
        return 0

    if not args.prereg.exists():
        print(f"error: no committed preregistration at {args.prereg}", file=sys.stderr)
        print("Run --phase prereg, commit the manifest, then dispatch.", file=sys.stderr)
        return 2
    manifest = json.loads(args.prereg.read_text(encoding="utf-8"))

    if args.phase == "nominate":
        rows = [
            FitRow(**json.loads(line))
            for path in sorted(args.out.glob("*.jsonl"))
            if "-contrasts" not in path.name and "-pilot" not in path.name
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        contrasts = contrast_rows(rows, seed=args.seed)
        decision = nominate(contrasts, rows)
        path = args.out / "nomination.json"
        path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(decision, indent=2, sort_keys=True))
        _emit(path.read_text(encoding="utf-8"), "NOMINATION")
        return 0

    if args.phase == "cost":
        print("error: --phase cost is not implemented yet", file=sys.stderr)
        return 2

    # --phase select
    per_size = {
        n: (
            seeds_for(manifest, args.cohort, n)
            if args.draws is None
            else seeds_for(manifest, args.cohort, n)[: args.draws]
        )
        for n in args.sizes
    }
    complaints = []
    for n, seeds in per_size.items():
        complaints.extend(
            validate_prereg(
                manifest,
                phase="select",
                cohort=args.cohort,
                seeds=seeds,
                draws_declared=len(seeds) if args.draws is not None else SIZE_DRAWS.get(n),
            )
        )
    if complaints:
        for line in dict.fromkeys(complaints):
            print(f"error: {line}", file=sys.stderr)
        return 1

    payloads = [
        Payload(
            cohort=args.cohort,
            cell=cell,
            n=n,
            data_seed=int(d),
            fold_seed=int(f),
            arms=tuple(a for a in ARMS if a in set(args.arms)),
            quadrature_points=args.quadrature_points,
            reference_points=args.reference_points,
        )
        for cell in args.cells
        for n in args.sizes
        for d, f in per_size[n]
    ]
    fits = sum(len(p.arms) for p in payloads)
    print(
        f"F5 phase 1: {len(payloads)} draws, {fits} fits, cohort={args.cohort}, "
        + ", ".join(f"n={n}: {len(s)} draws" for n, s in sorted(per_size.items()))
    )
    stamp = time.strftime("%Y%m%dT%H%M%S")
    rows = _run(payloads, args.jobs, args.out, f"{stamp}-{args.cohort}")
    contrasts = contrast_rows(rows, seed=args.seed)
    _write_rows(contrasts, args.out / f"{stamp}-{args.cohort}-contrasts.jsonl")

    print()
    print(_fit_table(rows))
    print()
    print(_contrast_table(contrasts, ("root_n_remaining", "score_failures")))
    _emit(
        "".join(json.dumps(asdict(row)) + "\n" for row in contrasts),
        f"CONTRASTS-{args.cohort.upper()}",
    )
    errored = sum(1 for row in rows if row.error)
    if errored:
        print(f"{errored} fit row(s) recorded an error", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
