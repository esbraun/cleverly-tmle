r"""F4's construction diagnostics: six paired contrasts, one factor each.

``docs/roadmap.md``'s piece F -- *Localize the shortfall before changing anything* --
puts **F4** fifth, beside F5, and this module is its instrument.  The shortfall
``docs/drtmle/coverage-study.md`` measured is not localized: it does not distinguish a
learner failure from the pooled reduced-crossfit construction, from the targeting/update/closing
construction, or from an interaction among them.  F4 is the **construction** half of that
question and F5 is the learner half; the two are independent, and either remains a theorem
premise even if the other succeeds.

**It closes nothing on its own and it may not branch, which is what it is for.**  Nothing under
``src/`` moves here -- only F7 may -- and F4's row forbids advancing a construction whose paired
change does not reproduce on the audit cohort, does not reduce ``sqrt(n) R_remaining`` or the
score-failure rate without worsening the other, or is not theorem-consistent.  **A null result
here is a result.**  Final coverage is in neither diagnostic; that is F8's and only F8's.

What the row asks for, and what this module is built to
-------------------------------------------------------

**A contrast that moves two things cannot say which of them moved the number.**  That is F3's
finding made concrete rather than a methodological preference: this package's two update orders
differ from R's round in *two* crossed ways -- the equation order and how many reduction vintages
a round adopts -- so ``"cleverly"`` against an R-style arm was never one factor.  Hence six
contrasts, each moving exactly one thing, all against a common reference arm except the second:

===========================  ==========================================  ======================
paired contrast              held fixed                                  factor isolated
===========================  ==========================================  ======================
``cleverly`` ~ ``r-style``   equation order ``9, 10, 8``; stopping;       reduction **vintage**
                             close; bounds; cross-fit                    ``all+all`` vs ``gr+qr``
``r-style`` ~ ``paper``      the two-vintage adoption; stopping;          **equation order**
                             close; bounds; cross-fit                    and refit placement
``cleverly`` ~ ``no-close``  route; stopping; bounds; cross-fit          the frozen-reduction
                                                                         **closing pass**
``cleverly`` ~ ``nested``    route; stopping; close; bounds              the reduction
                                                                         **cross-fitting**
``cleverly`` ~ ``loose``     one route; close; bounds; cross-fit         the **stopping rule**
===========================  ==========================================  ======================

The sixth factor -- the **truncation convention** -- is :func:`truncation_reading`'s, on two
frozen fixtures rather than over draws.  See below.

**Every arm is a benchmark-side construction and none is a production keyword.**  Two are shipped
diagnostic keywords already (``update_order="paper"``, ``reduced_crossfit="nested"``); the R-style
arm is :class:`~benchmarks.drtmle_trace.RStyleDRTMLE`; and the remaining three are **scoped**
patches of the module-level names in :mod:`cleverly.estimators.targeting`, installed for the
duration of one fit and restored in a ``finally`` -- the route
:class:`~benchmarks.drtmle_trace.TracingDRTMLE` already takes, and the only route to constants
``src/`` exposes through no keyword.

**The sixth contrast is read exactly and not over a cohort**, and that is a measurement rather
than a shortcut.  F4's row asks for the truncation convention in two declared designs -- a
bound-inactive theorem-side control and a bound-active stress design.  The tier-2 law is
``linear_dgp``, chosen for overlap rather than for difficulty, and its initial mechanism's clip
share is ``0.0000`` even at a bound of ``(0.15, 0.85)``: on this study's own draws the two arms
are **bit-identical**, so a cohort of them would report a null on a contrast that could not have
been non-null.  The two frozen trace fixtures already *are* the two regimes -- ``v1``'s bound is
slack on every row, ``v2``'s clips 54 of 200 -- so :func:`truncation_reading` answers the factor
on committed files, deterministically, at both ends of the alternation.  See
:data:`TRUNCATION_FIXTURES`.  Never a comparison built by conditioning on the realized post-fit
label: that is selecting the contrast on its own outcome, and F4's row forbids it by name.

The two phases, and why the freeze is a commit
----------------------------------------------

``--phase prereg`` writes the frozen manifest and **fits nothing**: the cohort seeds, the rule's
constants, the declared contrasts and the derived replicate count.  It is committed, and
``--phase run`` then :func:`validate_prereg`-checks the committed file before it fits anything --
a moved rule, a changed configuration, an overlapping cohort or an incomplete draw set is refused
with a non-zero exit rather than reported in a table nobody re-reads.

That is the standing decision E2R established, applied to a study that has no
data-dependent *selection* to freeze: what is frozen here is the **design**, so the manifest can
be produced in the environment the study is dispatched from rather than recovered from a first
job's log.

Usage
-----

.. code-block:: bash

    python -m benchmarks.drtmle_construction --phase prereg \
        --out evidence/f4-construction                        # writes prereg.json, no fits
    python -m benchmarks.drtmle_construction --phase run \
        --prereg evidence/f4-construction/prereg.json \
        --cells q-drift --sizes 600 --cohort selection --draws 2 --jobs 1   # a smoke run
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

if __package__ in {None, ""}:  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import drtmle_remainder, drtmle_tier2
from benchmarks.drtmle_trace import RStyleDRTMLE

from cleverly import DRTMLE
from cleverly.estimators import targeting as _targeting
from cleverly.estimators.base import format_table

__all__ = [
    "ARMS",
    "CONTRASTS",
    "PILOT_PAIRED_SPREAD",
    "SIZE_DRAWS",
    "TRUNCATION_FIXTURES",
    "ContrastRow",
    "FitRow",
    "arm_estimator",
    "cohort_seeds",
    "contrast_rows",
    "frozen_rule",
    "one_draw",
    "prereg",
    "raw_target",
    "replicate_count",
    "seeds_for",
    "truncation_reading",
    "validate_prereg",
    "write_prereg",
]

# ------------------------------------------------------------------ the frozen rule
#
# Every constant a verdict is read against, each with the reason it has the value it has.
# `frozen_rule()` serialises them into the manifest and `validate_prereg` refuses a run whose
# committed manifest disagrees, so the record and the rule cannot come apart.

#: The two misspecification regimes, and they are C3c's.  ``q-drift`` is the cell a coverage
#: shortfall is claimed in and ``g-drift`` is where the remainder is read off, so a construction
#: that moved one and not the other would be saying something this study has to be able to see.
CELLS = ("q-drift", "g-drift")

#: The two sizes.  F4's row names them, and they are E2R's rather than C3c's three: a contrast is
#: a **paired** difference within a draw, so the middle size buys a third reading of the same
#: comparison rather than a rate.
SIZES = (600, 2_400)

#: Tier 2 -- both nuisances fitted, the good one a smoother whose bandwidth sequence was
#: committed before any fit.  Tier 1 injects a prescribed sequence, which makes "the intended
#: asymptotic regime was entered" true by construction; the demonstration is tier 2 and so is
#: this diagnostic, because a construction effect read under an injected nuisance would be an
#: effect on an estimator nobody runs.
TIER = 2

#: The reduced learner, held at C3c's.  F4 is the **construction** half of the localization and
#: F5 is the learner half; a construction contrast taken at a learner F5 might move would be two
#: factors again, which is the mistake this whole matrix exists to remove.
REDUCED_LEARNER = "glm"

#: The primary estimand every contrast is read on.  ``ate`` because it is the parameter the
#: release criterion is about; ``ey1`` and ``ey0`` are recorded on every row and read as
#: secondary, never as three chances for one contrast to move.
PRIMARY_ESTIMAND = "ate"

#: The three verdicts, and the third is not a weak second.  ``moved`` needs the paired interval
#: to exclude zero **and** the effect to reproduce on the audit cohort; ``flat`` needs the
#: interval to lie inside :data:`NEGLIGIBLE_EFFECT`; anything else is ``unresolved``, which says
#: the study cannot tell and is a recorded outcome rather than a pass.
VERDICTS = ("moved", "flat", "unresolved")

#: What counts as a negligible paired effect on ``sqrt(n) R_remaining``, as a fraction of the
#: column C3c read.  ``0.10`` of a column sitting at ``1.17``--``1.25`` in ``q-drift`` is about
#: ``0.12``, and a construction that moves the remainder by less than a tenth of its standing
#: value has not localized a shortfall of that column's size.  **Declared before the first fit**
#: -- a margin chosen after a number exists is the failure mode stop-ship 17 names.
NEGLIGIBLE_EFFECT = 0.10

#: C3c's reading of ``sqrt(n) R_remaining`` in ``q-drift``, the column the margin above is a
#: fraction of.  Recorded here so the arithmetic is in the artefact rather than in a document.
C3C_REMAINING_QDRIFT = 1.25

#: E1b's **retained** across-draw spread of ``sqrt(n) R_remaining`` at ``n = 2,400`` under the
#: quadrature rule -- against ``2.05`` under C3c's draw rule, which is the sixfold shrink E1
#: bought.  Recorded because F4's row names it, and **not** used to size this study: it is a
#: spread of a *single arm's* column, and every outcome here is a **paired difference** whose
#: spread is a different quantity.  :data:`PILOT_PAIRED_SPREAD` is the one that sizes it.
E1B_RETAINED_SPREAD = 0.33

#: The **measured** paired spread of each contrast's ``sqrt(n) R_remaining``, per size, from a
#: 12-draw pilot on a **third seed stream disjoint from both cohorts** -- ``spawn(3)``'s child
#: 2, which leaves children 0 and 1 byte-identical to the two this study commits, so sizing
#: cannot move the cohorts it sizes.  Worst over the two cells.
#:
#: **Sizing on a measurement rather than on E1b's constant is the whole content of this table,
#: and the first two attempts at it were wrong in opposite directions.**  Taking E1b's ``0.33``
#: gave 27 draws, which is far too few at ``n = 600`` and more than enough at ``n = 2,400``.  A
#: 6-draw pilot then read the paired spreads at ``0.0001``--``1.36`` and implied 1 to 452
#: draws; at 12 draws the same quantities read ``0.0001``--``2.67``, so the 6-draw estimates
#: were low by a factor of 20 on three contrasts.  A spread estimated from six paired
#: differences is not an estimate of a spread.
#:
#: **The quadrature rule is not what drives it, which was checked rather than assumed.**  The
#: obvious suspect was the companion's own integration error -- E1's whole subject -- so the
#: pilot was run at 512 and at 2,048 points and the paired spreads moved by under 3%
#: (``2.5479`` against ``2.6717`` on the worst contrast).  What drives it is the *size*: at
#: ``n = 600`` the alternation's arms land at genuinely different fixed points draw by draw,
#: and by ``n = 2,400`` they do not.
PILOT_PAIRED_SPREAD: dict[int, dict[str, float]] = {
    600: {
        "r-style~cleverly": 0.0046,
        "paper~r-style": 2.6717,
        "no-close~cleverly": 1.1934,
        "nested~cleverly": 2.4506,
        "loose~cleverly": 2.3036,
    },
    2400: {
        "r-style~cleverly": 0.0001,
        "paper~r-style": 0.5188,
        "no-close~cleverly": 0.0067,
        "nested~cleverly": 0.4322,
        "loose~cleverly": 0.4710,
    },
}

#: How many draws each size takes, and the two are different **because the sizes answer
#: different questions at this margin**.  At ``n = 2,400`` the worst contrast needs 67 draws to
#: resolve the declared half-width, so 80 clears every contrast with margin.  At ``n = 600`` the
#: same arithmetic asks for 350 to 1,600 -- the spreads are five to twenty times larger -- which
#: is out of budget by an order of magnitude and would buy ``unresolved`` on every contrast that
#: matters.
#:
#: So ``n = 600`` runs at 24 draws and is **declared in advance as powered for ``moved`` and not
#: for ``flat``**: it can show a factor moves the column and it cannot show one does not.  That
#: is a stated limit of this study rather than a result of it, and every ``n = 600`` row carries
#: its realized minimum detectable effect so a reader is not left inferring the difference.
SIZE_DRAWS: dict[int, int] = {600: 24, 2400: 80}

#: The two-sided normal quantile the paired interval and :func:`replicate_count` are taken at.
Z_95 = 1.959963984540054

#: The bootstrap resample count for a paired interval, and the percentiles it is read at.
BOOTSTRAP = 2_000
INTERVAL = (2.5, 97.5)

#: A run is complete when it holds at least this share of its declared draws.  Below it the
#: cell is ``unresolved`` **by rule** rather than reported on whatever survived -- a study that
#: quietly shrank to the draws that worked would report a selected sample as a full one.
COMPLETENESS_FRACTION = 0.9

#: The quadrature rule ``P_0 D-hat`` is integrated on, and the scramble count per draw.  E1b is
#: why the scramble is independent per replicate rather than one fixed grid: a fixed grid's error
#: is a bias no replicate count removes, and randomising makes it mean-zero instead.
QUADRATURE_POINTS = 2_048
QUADRATURE_SCRAMBLES = 2

#: The score-check tolerance a fit's validity is read at, imported rather than restated.
VALIDITY_TOLERANCE = 1e-3

#: The seed-stream family.  ``90``--``92M`` are C3c's and E1b's and ``103``--``106M`` are E2R's,
#: so F4 takes a family nothing else has used: two draws sharing a stream would be the same rows
#: under two headings.
COHORT_SEED = 20250801
QUADRATURE_SEED = 110_000_000

#: The two cohorts, in the order they are read.  A cohort is a **disjoint set of simulation
#: draws**: the contrast is read on the selection cohort and has to *reproduce* on the audit
#: cohort, which is the whole content of F4's acceptance clause.
COHORTS = ("selection", "audit")

#: A **third** child of the same seed sequence, reserved for the sizing pilot
#: :data:`PILOT_PAIRED_SPREAD` was measured on, and spent.  It is named here rather than left
#: implicit because a study sized on draws it then reads would be sizing on its own outcome.
#:
#: ``SeedSequence.spawn`` gives child ``i`` the same state whatever ``n`` is, so reserving a
#: third child leaves the two cohorts byte-identical to what ``spawn(2)`` produced.  That is a
#: property this study depends on and ``tests/unit/test_drtmle_construction.py`` checks it.
SIZING_STREAM = 2


def frozen_rule() -> dict[str, Any]:
    """Every constant a verdict is read against, as a dictionary a manifest can carry."""
    return {
        "bootstrap": BOOTSTRAP,
        "c3c_remaining_qdrift": C3C_REMAINING_QDRIFT,
        "completeness_fraction": COMPLETENESS_FRACTION,
        "e1b_retained_spread": E1B_RETAINED_SPREAD,
        "interval": list(INTERVAL),
        "negligible_effect": NEGLIGIBLE_EFFECT,
        "primary_estimand": PRIMARY_ESTIMAND,
        "validity_tolerance": VALIDITY_TOLERANCE,
        "verdicts": list(VERDICTS),
    }


# ------------------------------------------------------------------ the replicate count
#
# Derived, printed and committed -- never a literal. F4's row asks for a count "sized from C3c's
# and E1b's **retained** variability rather than from an invented replicate count", and the
# distinction is that the arithmetic below is in the artefact.


def minimum_detectable_effect(spread: float, draws: int) -> float:
    """The smallest paired effect this many draws can separate from zero, at 95%."""
    return float(Z_95 * spread / math.sqrt(draws)) if draws > 0 else float("inf")


def replicate_count(half_width: float | None = None) -> dict[str, Any]:
    r"""How many draws each size takes, and the arithmetic that says so.

    The target half-width is :data:`NEGLIGIBLE_EFFECT` of the column C3c read, so the study is
    sized to resolve exactly the effect the margin calls negligible -- a study that could not
    tell a negligible effect from a material one would return ``unresolved`` by construction.

    Per **size**, because :data:`PILOT_PAIRED_SPREAD` says the two sizes are five to twenty
    times apart on every contrast that matters, and one count for both would be either
    unaffordable or uninformative.  The returned ``required`` is what the pilot's worst contrast
    asks for and ``draws`` is what the study commits; where they differ, ``powered`` says which
    verdicts the committed count supports, so a reader never has to infer it from a wide
    interval.
    """
    target = NEGLIGIBLE_EFFECT * C3C_REMAINING_QDRIFT if half_width is None else half_width
    out: dict[str, Any] = {"half_width": float(target), "sizes": {}}
    for size, spreads in PILOT_PAIRED_SPREAD.items():
        worst_contrast = max(spreads, key=lambda name: spreads[name])
        worst = spreads[worst_contrast]
        required = math.ceil((Z_95 * worst / target) ** 2)
        committed = SIZE_DRAWS[size]
        out["sizes"][str(size)] = {
            "committed": committed,
            "mde": {
                name: round(minimum_detectable_effect(sd, committed), 4)
                for name, sd in spreads.items()
            },
            "powered": "moved and flat" if committed >= required else "moved only",
            "required": required,
            "worst_contrast": worst_contrast,
            "worst_spread": worst,
            "arithmetic": (
                f"worst paired sd {worst} ({worst_contrast}); "
                f"ceil(({Z_95:.3f} * {worst} / {target:.4f})^2) = {required}; "
                f"committed {committed}"
            ),
        }
    return out


def cohort_seeds(seed: int, draws: int) -> dict[str, list[tuple[int, int]]]:
    """One ``(data_seed, fold_seed)`` list per cohort, from **disjoint** streams.

    ``SeedSequence(seed).spawn(2)``, one child per cohort, rather than one stream sliced in two.
    A slice is prefix-stable -- ``docs/drtmle/study-manifest.md`` records C3c running into
    exactly that, where a "fresh" batch turned out to share the pilot's data seeds -- so raising
    one cohort's count would shift which draws the other took.

    **The disjointness is the experiment and not housekeeping.**  A contrast read across the
    selection cohort and reproduced on draws that cohort contained would be assessing an effect
    on the sample that produced it.  :func:`validate_prereg` checks it again at the run, on the
    **data** seed, because two draws sharing one under different splits are the same rows twice.
    """
    children = np.random.SeedSequence(seed).spawn(SIZING_STREAM + 1)
    out: dict[str, list[tuple[int, int]]] = {}
    for name, child in zip(COHORTS, children, strict=False):
        state = child.generate_state(2 * draws)
        out[name] = [
            (int(data), int(fold)) for data, fold in zip(state[:draws], state[draws:], strict=True)
        ]
    return out


def seeds_for(manifest: Mapping[str, Any], cohort: str, size: int) -> list[tuple[int, int]]:
    """The draws one ``(cohort, size)`` runs, as a **prefix** of the committed cohort.

    A prefix rather than a fresh stream per size, so that ``n = 600``'s draws are a subset of
    ``n = 2,400``'s: the two sizes then differ in the size and in nothing else, which is what
    lets a reader compare a contrast across them without wondering whether the draws moved.
    """
    declared = [(int(a), int(b)) for a, b in manifest["cohorts"][cohort]]
    return declared[: SIZE_DRAWS.get(size, len(declared))]


# ------------------------------------------------------------------ the arms
#
# Three of the seven are scoped patches of module-level names, which is the only route to
# constants `src/` exposes through no keyword -- `max_outer`, `_NEGLIGIBLE`, and the fact that
# the closing pass always runs. Scoped to one fit and restored in a `finally`, so an ordinary
# `DRTMLE` in the same process is untouched and a raise inside the alternation cannot leave the
# module patched. `tests/unit/test_drtmle_construction.py` checks both rather than arguing them.


@contextmanager
def _patched(**names: Any) -> Iterator[None]:
    """Install module-level names on :mod:`cleverly.estimators.targeting`, then restore them."""
    original = {name: getattr(_targeting, name) for name in names}
    for name, value in names.items():
        setattr(_targeting, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(_targeting, name, value)


def _identity_close(
    data: Any,
    nuisance: Any,
    group: Any,
    spec: Any,
    *,
    reduced: Any,
    guard: tuple[str, ...],
    bounds: tuple[float, float],
    nuisance_bound: float,
    scaled: Any,
    weights: Any,
    observed: Any,
    mask: Any,
    indicator: Any,
    arms: tuple[float, ...],
    targeted_g: Any,
    fluctuation: Any,
    mechanism: Any,
    extra: Any,
    companion: Any = None,
    max_steps: int = 20,
) -> Any:
    """``_close_at_frozen_reductions`` with the pass removed and the state handed straight back.

    **The pre-close state is reachable from nowhere on a returned fit.**  The closing pass runs
    unconditionally and ``ReductionFluctuation`` carries only post-close arrays, so reading the
    boundary means either recording it -- which is F2's trace, and gives a state rather than a
    result object -- or removing the pass, which is this.  Removing it is what F4's contrast
    needs: every column the matrix reports comes off a **result**, so the two sides of this
    contrast have to be two results and not a result and a state.

    Nothing is recomputed.  The loop has already restated all three scores at the pair the round
    exits at -- equation (8) through ``_restated_outcome_score``, equation (10) at the final
    reductions, equation (9) at the exiting mechanism -- so the incoming ``fluctuation``,
    ``extra`` and ``mechanism`` are already the pre-close record.  What is rebuilt is only the
    two submodels, at exactly the expressions the real pass builds them at, because they are
    what the caller reads the estimate off.

    ``steps=0`` and ``capped=False`` are the honest record of a pass that did not run, and they
    are what makes the arm legible in the artefact: a ``closing`` of ``0`` beside the reference
    arm's is how a reader tells this row apart without consulting the harness.
    """
    current = (
        _targeting._retargeted_mechanism(nuisance, targeted_g, arms) if "Q" in guard else nuisance
    )
    submodel = _targeting.build_submodel(
        data, current, group, bounds=bounds, nuisance_bound=nuisance_bound
    )
    extra_submodel = (
        _targeting.reduced_outcome_submodel(data.treatment, reduced, bounds=bounds)
        if "g" in guard and extra is not None
        else None
    )
    reduced_score = (
        0.0 if extra is None else float(_targeting.relative_score(extra.score, extra.score_scale))
    )
    return _targeting._Closing(
        submodel=submodel,
        fluctuation=fluctuation,
        mechanism=mechanism,
        extra=extra,
        extra_submodel=extra_submodel,
        reduced_score=reduced_score,
        mechanism_score=0.0 if mechanism is None else mechanism.relative_score,
        joint=float(fluctuation.loglik)
        + float(0.0 if mechanism is None else (mechanism.loglik or 0.0)),
        steps=0,
        capped=False,
    )


#: Wide enough to be inert on a probability, and the value
#: ``tests/unit/test_reduction_alternation.py`` uses for the same purpose.
INERT_BOUNDS = (0.0, 1.0)


def raw_target(estimator: DRTMLE) -> DRTMLE:
    r"""Form :math:`g_{r,2}`'s target at the **untruncated** mechanism, on any estimator.

    The truncation convention, isolated.  :math:`g_{r,2}`'s target is
    :math:`(1_a - \hat g)/\hat g` and the :math:`\hat g` in it is the *truncated* one -- the one
    bound in this package chosen at fit time rather than at targeting time, recorded on
    ``ReducedSet.g_bounds`` precisely so a reader of a truncation curve can find out the sweep
    did not reach these arrays.  This forms it at the raw mechanism instead.

    **It moves that one array and nothing else.**  The covariate denominators, ``g_{r,1}``'s
    read-time clip and the mechanism the loop carries forward are all untouched, because they
    are bounded at targeting time from ``bounds`` and this reaches only ``fit_reduced``'s
    ``g_bounds`` argument.  A ``g_bounds=`` change on the constructor would move all of them at
    once, which is the bundled arm F4's row forbids.

    **A function on an instance rather than a subclass, and that is the point.**  This factor is
    needed on two different estimators -- the tier-2 one an arm is built from, and the frozen
    fixture's one :func:`truncation_reading` fits -- and it was written twice, as a class and as
    an inline closure.  A mutation removing the factor from the class then left the truncation
    reading green, because the reading never used the class: two implementations of one factor,
    with the drift between them invisible.  One function, called from both.
    """
    inner = estimator._fit_reduced

    def _fit_reduced(data: Any, nuisance: Any, _g_bounds: Any) -> Any:
        return inner(data, nuisance, INERT_BOUNDS)

    estimator._fit_reduced = _fit_reduced  # type: ignore[method-assign]
    return estimator


#: The arms, and what each one is.  ``factor`` is what a contrast against ``base`` isolates, and
#: ``base`` is the arm it is read against -- ``r-style`` is the only one whose base is not the
#: shipped reference, because the equation-order contrast is R's round against the paper's and
#: neither of those is ``cleverly``.
ARMS: dict[str, dict[str, Any]] = {
    "cleverly": {"factor": None, "base": None, "why": "the shipped alternation"},
    "r-style": {
        "factor": "reduction vintage",
        "base": "cleverly",
        "why": "R's round: R's equation order with R's two reduction vintages",
    },
    "paper": {
        "factor": "equation order and refit placement",
        "base": "r-style",
        "why": "the working paper's six-step order, with the same two vintages",
    },
    "no-close": {
        "factor": "the frozen-reduction closing pass",
        "base": "cleverly",
        "why": "the state the alternation exits at, before the closing pass",
    },
    "nested": {
        "factor": "the reduction cross-fitting construction",
        "base": "cleverly",
        "why": "fold k's reductions trained on models that left fold k out as well",
    },
    "loose": {
        "factor": "the stopping rule",
        "base": "cleverly",
        "why": "R's tolIC = 1/n in place of this package's 1e-3/n",
    },
}

#: The truncation arm, kept out of :data:`ARMS` **on a measurement**.  It is F4's sixth factor
#: and it is read by :func:`truncation_reading` on the two frozen fixtures rather than by a
#: cohort of draws -- see :data:`TRUNCATION_FIXTURES`.  It stays a class because that reading
#: needs it, and it is named here so nothing has to reach into the module to find it.
TRUNCATION_ARM = "raw"

#: The six paired contrasts, in the order they are read.  Declared here rather than derived from
#: :data:`ARMS` so that the reading order is itself frozen: F4's row asks for "one primary
#: contrast per factor **and the order they are read in**", and an order that fell out of a dict
#: iteration would move the moment an arm was added.
CONTRASTS = tuple((name, spec["base"]) for name, spec in ARMS.items() if spec["base"])

#: The truncation contrast's two declared designs, and **they are frozen fixtures rather than
#: cohorts of draws**.  F4's row asks for a bound-inactive theorem-side control and a
#: bound-active stress design, each declared in advance; ``benchmarks/fixtures/drtmle_trace_v1``
#: and ``_v2`` already **are** those two regimes -- ``v1``'s bound is slack on every row and
#: ``v2``'s clips 54 of 200 -- and they are frozen files, so "declared in advance" is literal
#: rather than a promise about a seed.
#:
#: **This is an exact reading and not a cohort study, and the reason is a measurement.**  The
#: tier-2 law is ``linear_dgp``, chosen for overlap rather than for difficulty, and its initial
#: mechanism's clip share is ``0.0000`` even at a bound of ``(0.15, 0.85)`` -- so on the study's
#: own draws the two arms are **bit-identical** and a cohort of them would report a null on a
#: contrast that could not have been non-null.  A declared stress design that cannot be
#: stressed is worse than no stress design: it spends fits and reads as evidence.
#: ``CLAUDE.md``'s own order of preference puts an exact identity above a simulation, and this
#: contrast is one of the places that rule pays.
TRUNCATION_FIXTURES = {
    "v1": "the bound is slack on every row -- the theorem-side control",
    "v2": "the bound clips 54 of 200 rows -- the stress design, outside section 7's scope",
}


def arm_estimator(arm: str, settings: Mapping[str, Any], **kwargs: Any) -> DRTMLE:
    """The estimator for one arm, at the shared tier-2 settings.

    Every arm that is not a scoped patch is a class or a shipped diagnostic keyword, so this
    returns an ordinary estimator and :func:`_run_arm` decides whether a patch is in force.
    Keeping the two apart is what lets a test build an arm and inspect it without fitting.
    """
    # `TRUNCATION_ARM` is buildable and is deliberately not in `ARMS`: it is F4's sixth factor
    # and it is read by `truncation_reading` on two frozen fixtures rather than over a cohort.
    if arm not in ARMS and arm != TRUNCATION_ARM:
        raise ValueError(f"unknown arm {arm!r}; choose from {sorted(ARMS)} or {TRUNCATION_ARM!r}")
    shared = {
        **settings,
        "reduced_outcome_learner": REDUCED_LEARNER,
        "reduced_treatment_learner": REDUCED_LEARNER,
        **kwargs,
    }
    if arm == "r-style":
        return RStyleDRTMLE(**shared)
    if arm == TRUNCATION_ARM:
        return raw_target(DRTMLE(**shared))
    if arm == "paper":
        return DRTMLE(**shared, update_order="paper")
    if arm == "nested":
        return DRTMLE(**shared, reduced_crossfit="nested")
    return DRTMLE(**shared)


@contextmanager
def _arm_patch(arm: str) -> Iterator[None]:
    """Whatever module-level patch the arm needs, for the duration of one fit."""
    if arm == "no-close":
        with _patched(_close_at_frozen_reductions=_identity_close):
            yield
    elif arm == "loose":
        # `_negligible_bar(n)` is `_NEGLIGIBLE / n`, so this is R's `tolIC = 1/n` exactly. F3
        # measured the two rules three orders apart at n=200; this arm is what asks whether that
        # gap is a difference the reported columns can see.
        with _patched(_NEGLIGIBLE=1.0):
            yield
    else:
        yield


# ------------------------------------------------------------------ the truncation contrast
#
# F4's sixth factor, read exactly on two frozen fixtures rather than over a cohort of draws --
# see TRUNCATION_FIXTURES for the measurement that decided that.


@dataclass(frozen=True)
class TruncationRow:
    """The truncation convention on one frozen fixture, at both ends of the alternation."""

    fixture: str
    clipped: int
    rows: int
    stage: str
    quantity: str
    identical: bool
    worst: float


def truncation_reading() -> list[TruncationRow]:
    r"""Does forming :math:`g_{r,2}`'s target at the raw mechanism change anything?

    Read at **two** stages, and the pair is the whole finding: at the *initial* reduced fit,
    which is where the two conventions differ by construction, and at the *converged* set the
    reported curve actually reads.

    **Why the second stage is not redundant.**  ``solve_with_reduction`` carries the
    **truncated** tilt forward -- item 20's fix, ``targeting.py``'s "the truncated tilt, which
    is what makes the next round's offset, every later covariate and the reported correction
    read one array".  So from the first mechanism update onward :math:`g^*` already lies inside
    the covariate bounds, and re-truncating it for a reduction target is a no-op.  The
    convention can therefore differ at the initial fit and agree at every refit after it, which
    is exactly what this reads out -- and it is invisible to any comparison that looks only at
    a fitted result.

    No cohort, no seeds, no sampling: both fixtures are committed files and both arms are
    deterministic, so this is an identity rather than an estimate.
    """
    from benchmarks import drtmle_trace as trace_module

    rows: list[TruncationRow] = []
    for version in TRUNCATION_FIXTURES:
        fixture = trace_module.read_fixture(version=version)
        clipped, total = int(fixture.manifest["clipped"]), int(fixture.manifest["n"])
        produced = {}
        for label, raw in (("bounded", False), ("raw", True)):
            estimator = trace_module.estimator(order="cleverly", tracing=False, version=version)
            if raw:
                raw_target(estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = trace_module._fit(fixture.frame, estimator)
            produced[label] = {
                "initial": fit.nuisance.reduced,
                "converged": fit.fluctuations["mean"].reduction.reduced,
            }
        for stage in ("initial", "converged"):
            for quantity in ("qr", "gr1", "gr2"):
                left = np.asarray(getattr(produced["bounded"][stage], quantity), dtype=float)
                right = np.asarray(getattr(produced["raw"][stage], quantity), dtype=float)
                rows.append(
                    TruncationRow(
                        fixture=version,
                        clipped=clipped,
                        rows=total,
                        stage=stage,
                        quantity=quantity,
                        identical=bool(np.array_equal(left, right)),
                        worst=float(np.max(np.abs(left - right))),
                    )
                )
    return rows


TRUNCATION_HEADERS = ("fixture", "clipped", "stage", "quantity", "identical", "worst |diff|")


def _truncation_table(rows: Sequence[TruncationRow]) -> str:
    return format_table(
        TRUNCATION_HEADERS,
        [
            (
                row.fixture,
                f"{row.clipped}/{row.rows}",
                row.stage,
                row.quantity,
                "yes" if row.identical else "no",
                f"{row.worst:.4e}",
            )
            for row in rows
        ],
    )


# ------------------------------------------------------------------ the rows


@dataclass(frozen=True)
class FitRow:
    """One arm on one draw, flat and JSON-serialisable so an artefact needs no schema to read.

    ``root_n_remaining`` and ``score_failures`` are the two **primary** columns; everything else
    is a secondary diagnostic F4's row names.  ``psi`` and ``std_error`` are here because the
    point-estimate movement and the ``se`` ratio are differences of them, and a difference
    computed in the artefact is one a reader can recompute.
    """

    cohort: str
    cell: str
    n: int
    data_seed: int
    fold_seed: int
    arm: str
    estimand: str
    psi: float
    truth: float
    std_error: float
    root_n_remaining: float
    score_8: float
    score_9: float
    score_10: float
    reduction_drift: float
    identity_failures: int
    score_failures: int
    valid: bool
    rounds: int
    closing: int
    exit_reason: str
    failure: str
    bound_active: bool
    initial_clip_share: float
    seconds: float
    error: str = ""


@dataclass(frozen=True)
class ContrastRow:
    """One paired contrast in one cell, on one cohort -- the grain a verdict is read at."""

    cohort: str
    cell: str
    n: int
    contrast: str
    factor: str
    column: str
    draws: int
    mean: float
    lower: float
    upper: float
    paired_sd: float
    verdict: str


def _failure_counts(check: Any) -> tuple[int, int]:
    """A failing fit's two causes, counted apart, the way C3c's gate 1 reads them."""
    identity = len(check.identity_failures)
    return identity, len(check.failures) - identity


def _scores(reduction: Any, fluctuation: Any) -> tuple[float, float, float]:
    """The three empirical means at exit, as the loop's own trace last recorded them.

    ``reduction.trace`` carries one row per round plus one for the closing pass, each
    ``(round, equation-8, equation-10, equation-9, joint)``, so the last row is the state the
    fit reports at -- which is the state Theorem 1's premise is about and the state the reported
    curve is built from.
    """
    if reduction is None or not reduction.trace:
        return (float(fluctuation.relative_score_norm), 0.0, 0.0)
    last = reduction.trace[-1]
    return (float(last[1]), float(last[3]), float(last[2]))


def _reduction_drift(fit: Any, data: Any, estimator: DRTMLE) -> float:
    r"""How far the reported reductions are from a refit at the state the fit reports at.

    F4's row asks for "the reduced regressions' drift between the last refit and the reported
    state", and this is that quantity rather than a proxy for it: the reductions the curve reads
    were fitted at a pair the loop has since moved, so refitting once **at the exit pair** and
    differencing says how stale they are.  Reported as the worst absolute difference over the
    three regressions, scaled by the pooled spread of the two sides so a small number on a small
    array is not read as agreement.

    It costs one reduction fit per arm per draw and no alternation.
    """
    fluctuation = fit.fluctuations["mean"]
    reduction = fluctuation.reduction
    if reduction is None:
        return float("nan")
    nuisance = fit.nuisance
    targeted_g = nuisance.propensity
    if fluctuation.mechanism is not None:
        targeted_g = _targeting._propensity_from(
            fluctuation.mechanism.propensity, tuple(nuisance.arms)
        )
    at_exit = replace(nuisance, propensity=targeted_g, outcome=fluctuation.targeted)
    fresh, _, _ = estimator._fit_reduced(data, at_exit, reduction.reduced.g_bounds)
    worst = 0.0
    for name in ("qr", "gr1", "gr2"):
        theirs = np.asarray(getattr(reduction.reduced, name), dtype=float)
        ours = np.asarray(getattr(fresh, name), dtype=float)
        spread = float(np.std(np.concatenate([theirs.reshape(-1), ours.reshape(-1)])))
        gap = float(np.max(np.abs(theirs - ours)))
        worst = max(worst, gap / spread if spread > 0 else gap)
    return worst


def _bound_witness(fit: Any) -> tuple[bool, float]:
    """Whether the truncation was active at exit, and the initial mechanism's clip share.

    Recorded on **every** row and never used to select or stratify a primary result -- F4's row
    forbids conditioning a contrast on the realized post-fit label.  It is here so a reader can
    check the regime the study actually ran in, and it is what says the tier-2 law never reaches
    a bound-active one: measured at ``0.0000`` on every draw, which is why the truncation
    contrast is :func:`truncation_reading`'s and not a column here.
    """
    try:
        check = fit.validation.score_check(tolerance=VALIDITY_TOLERANCE)
        clipped = max((int(row.clipped) for row in check.rows), default=0)
    except Exception:  # pragma: no cover - a diagnostic must not fail a fit
        clipped = 0
    # Every arm's column, not just the upper one: the share is of rows the bound reached at
    # *any* arm, which is what `initial clip share` means in C3c's table.
    propensity = np.asarray(fit.nuisance.propensity.values, dtype=float)
    lower, upper = fit.fluctuations["mean"].reduction.bounds
    outside = (propensity < lower) | (propensity > upper)
    share = float(np.mean(outside.any(axis=1))) if outside.ndim > 1 else float(np.mean(outside))
    return bool(clipped > 0), share


# ------------------------------------------------------------------ one draw


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


def _companion(payload: Payload, dgp: Any) -> Any:
    """The quasi-random evaluation rule, with an independent scramble per draw.

    E1b is why this is the rule rather than a fixed grid, and why the scramble moves: a fixed
    grid's error is a bias no replicate count removes, while a randomised rule is unbiased at
    every point count, so the error is mean-zero and averages down over the cohort.
    """
    first = QUADRATURE_SEED + payload.data_seed % 1_000_003
    scrambles = tuple(first + i for i in range(QUADRATURE_SCRAMBLES))
    return drtmle_remainder.stacked_companion(
        dgp, points=payload.quadrature_points, scrambles=scrambles
    ), scrambles


def one_draw(payload: Payload) -> list[FitRow]:
    """Every arm on one draw, paired inside the worker.

    **Paired inside the worker rather than across passes**, which is what makes every contrast a
    paired quantity: the arms see the same rows, the same fold seed and the same tier-2 nuisance
    specification, so every difference between two of them is the one factor that separates them.

    An arm that raises is **recorded** rather than dropped, one row per estimand with its error
    named.  Dropping it would condition the contrast on the draws where both arms happened to
    work, and an arm that raises more often than its base is itself a finding.
    """
    dgp = drtmle_tier2.base_law()
    frame, _ = dgp.sample(payload.n, seed=payload.data_seed)
    truth = dgp.truth()
    settings = dict(drtmle_tier2.settings(payload.cell, payload.n))

    stack, _ = _companion(payload, dgp)
    truths = [
        drtmle_remainder.truth_at(dgp, payload.quadrature_points, scramble=block.seed)
        for block in stack.blocks
    ]
    windows = [block.window for block in stack.blocks]

    rows: list[FitRow] = []
    for arm in payload.arms:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator = arm_estimator(
                    arm,
                    settings,
                    random_state=payload.fold_seed,
                    evaluation=stack.frame,
                )
                with _arm_patch(arm):
                    result = estimator.fit(frame, outcome="Y", treatment="A")
                fit = result.single()
                data = fit.data
                remainder = drtmle_remainder.remainder_rows(
                    fit,
                    dgp,
                    # The **fitting** size, not the companion's: the companion is a quadrature
                    # rule and its row count is an accuracy knob rather than a sample size any
                    # root-n scaling is stated in.
                    n=payload.n,
                    bounds=fit.config.g_bounds,
                    row_weights=stack.weights,
                    windows=windows,
                    truths=truths,
                )
                drift = _reduction_drift(fit, data, estimator)
                # Inside the guard with the fit itself, and that is the point: a post-fit read
                # that raises is a row this harness has to *record*, exactly as a fit that
                # raises is. Outside it, one arm's missing attribute aborted the whole
                # dispatch and produced no artefact at all -- measured, before this moved.
                fluctuation = fit.fluctuations["mean"]
                reduction = fluctuation.reduction
                check = fit.validation.score_check(tolerance=VALIDITY_TOLERANCE)
                identity_failures, score_failures = _failure_counts(check)
                score_8, score_9, score_10 = _scores(reduction, fluctuation)
                active, share = _bound_witness(fit)
        except Exception as exc:  # recorded and reported, never swallowed
            rows.extend(_failed(payload, arm, f"{type(exc).__name__}: {exc}", truth))
            continue
        seconds = time.perf_counter() - started
        by_estimand = {row.estimand: row for row in remainder}
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
                    std_error=float(estimate.std_error),
                    root_n_remaining=float("nan") if row is None else float(row.root_n_remaining),
                    score_8=score_8,
                    score_9=score_9,
                    score_10=score_10,
                    reduction_drift=drift,
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
            std_error=float("nan"),
            root_n_remaining=float("nan"),
            score_8=float("nan"),
            score_9=float("nan"),
            score_10=float("nan"),
            reduction_drift=float("nan"),
            identity_failures=0,
            score_failures=0,
            valid=False,
            rounds=0,
            closing=0,
            exit_reason="",
            failure="",
            bound_active=False,
            initial_clip_share=float("nan"),
            seconds=float("nan"),
            error=error,
        )
        for name in ("ate", "ey1", "ey0")
    ]


# ------------------------------------------------------------------ the contrasts


def _paired_interval(values: np.ndarray, seed: int) -> tuple[float, float, float, float]:
    """A paired difference's mean, its percentile interval and its spread.

    A bootstrap over draws rather than a normal interval, because the paired differences of
    ``sqrt(n) R_remaining`` are not obviously symmetric and a study this small cannot check.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(finite, size=(BOOTSTRAP, finite.size), replace=True).mean(axis=1)
    lower, upper = np.percentile(draws, INTERVAL)
    return (float(finite.mean()), float(lower), float(upper), float(finite.std(ddof=1)))


def _verdict(lower: float, upper: float, negligible: float) -> str:
    """``moved``, ``flat`` or ``unresolved`` -- and the third is not a weak second."""
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return "unresolved"
    if lower > 0.0 or upper < 0.0:
        return "moved"
    if -negligible <= lower and upper <= negligible:
        return "flat"
    return "unresolved"


#: Which columns a contrast is read on.  The first two are F4's **primary** outcomes and the
#: rest are the secondary diagnostics its row names; a verdict is taken on the primaries and the
#: rest are read beside them.
CONTRAST_COLUMNS = (
    "root_n_remaining",
    "score_failures",
    "psi",
    "std_error",
    "reduction_drift",
    "rounds",
)


def contrast_rows(rows: Sequence[FitRow], *, seed: int = COHORT_SEED) -> list[ContrastRow]:
    """Every declared contrast, on every cell, at the primary estimand.

    Read at :data:`PRIMARY_ESTIMAND` and nowhere else: three estimands would be three chances
    for one contrast to move, and F4's row asks for **one** primary contrast per factor.
    """
    indexed: dict[tuple[Any, ...], FitRow] = {}
    for row in rows:
        if row.estimand != PRIMARY_ESTIMAND:
            continue
        indexed[(row.cohort, row.cell, row.n, row.data_seed, row.arm)] = row

    cells = sorted({(r.cohort, r.cell, r.n) for r in rows})
    negligible = NEGLIGIBLE_EFFECT * C3C_REMAINING_QDRIFT
    out: list[ContrastRow] = []
    for cohort, cell, n in cells:
        seeds = sorted({r.data_seed for r in rows if (r.cohort, r.cell, r.n) == (cohort, cell, n)})
        for arm, base in CONTRASTS:
            for column in CONTRAST_COLUMNS:
                paired = []
                for data_seed in seeds:
                    left = indexed.get((cohort, cell, n, data_seed, arm))
                    right = indexed.get((cohort, cell, n, data_seed, base))
                    if left is None or right is None or left.error or right.error:
                        continue
                    paired.append(float(getattr(left, column)) - float(getattr(right, column)))
                mean, lower, upper, sd = _paired_interval(np.asarray(paired, dtype=float), seed)
                out.append(
                    ContrastRow(
                        cohort=cohort,
                        cell=cell,
                        n=n,
                        contrast=f"{arm}~{base}",
                        factor=str(ARMS[arm]["factor"]),
                        column=column,
                        draws=len(paired),
                        mean=mean,
                        lower=lower,
                        upper=upper,
                        paired_sd=sd,
                        verdict=_verdict(lower, upper, negligible)
                        if column == "root_n_remaining"
                        else _verdict(lower, upper, float("inf")),
                    )
                )
    return out


# ------------------------------------------------------------------ the frozen manifest


def prereg(
    *,
    seed: int = COHORT_SEED,
    cells: Sequence[str] = CELLS,
    sizes: Sequence[int] = SIZES,
    draws: int | None = None,
) -> dict[str, Any]:
    """The whole frozen design, as a dictionary a commit can carry.

    Four keys, and each answers a different question a reader of a verdict has: ``rule`` is
    every constant the verdict is read against, ``configuration`` is what the instrument was
    built at, ``cohorts`` is which draws each half is, and ``contrasts`` is what was declared to
    be compared and in what order.
    """
    sizing = replicate_count()
    count = max(SIZE_DRAWS.values()) if draws is None else int(draws)
    return {
        "cohorts": {
            name: [list(pair) for pair in pairs]
            for name, pairs in cohort_seeds(seed, count).items()
        },
        "configuration": {
            "arms": sorted(ARMS),
            "cells": list(cells),
            "draws": count,
            "draws_by_size": {str(k): v for k, v in sorted(SIZE_DRAWS.items())},
            "quadrature_points": QUADRATURE_POINTS,
            "quadrature_scrambles": QUADRATURE_SCRAMBLES,
            "reduced_learner": REDUCED_LEARNER,
            "seed": seed,
            "sizes": list(sizes),
            "tier": TIER,
        },
        "contrasts": [
            {
                "arm": arm,
                "base": base,
                "columns": list(CONTRAST_COLUMNS),
                "factor": ARMS[arm]["factor"],
                "order": index,
            }
            for index, (arm, base) in enumerate(CONTRASTS, start=1)
        ],
        "rule": {**frozen_rule(), "sizing": sizing},
    }


def write_prereg(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write the manifest sorted and indented, so a commit of it is reviewable line by line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


#: The configuration keys a run may not differ from the committed manifest on.  ``draws`` is
#: **not** among them: a smoke run takes fewer, and the completeness clause is what decides
#: whether a cell is readable rather than a strict equality that would forbid a debug run.
PINNED_CONFIGURATION = (
    "arms",
    "quadrature_points",
    "quadrature_scrambles",
    "reduced_learner",
    "seed",
    "tier",
)


def validate_prereg(
    manifest: Mapping[str, Any],
    *,
    cohort: str,
    seeds: Sequence[tuple[int, int]],
    draws_declared: int | None = None,
) -> list[str]:
    """Every reason this run may not be read against that manifest.  Empty is the only pass.

    Checked **before anything is fitted**, and each clause is a way a study can look like it
    answered its question while answering another one:

    * a **moved rule** -- a threshold that changed after the freeze is stop-ship 17's failure
      mode, and a verdict read against it is a verdict read against a bar chosen to clear;
    * a **changed configuration** -- an instrument built at other settings answers a different
      question under the same headings;
    * an **overlapping cohort** -- checked on the **data** seed, since two draws sharing one
      under different splits are the same rows twice, and an effect reproduced on the draws that
      produced it has not reproduced;
    * an **unknown cohort**, so a typo cannot silently produce a third cohort nobody declared;
    * an **incomplete draw set** -- below :data:`COMPLETENESS_FRACTION` the cell is unresolved by
      rule rather than reported on whatever survived.
    """
    complaints: list[str] = []
    if manifest.get("rule", {}).get("bootstrap") is None:
        complaints.append("rule: the manifest carries no rule block")
    else:
        frozen = frozen_rule()
        recorded = {key: manifest["rule"].get(key) for key in frozen}
        for key, value in frozen.items():
            if recorded.get(key) != value:
                complaints.append(
                    f"rule: {key} is {value!r} here and {recorded.get(key)!r} in the manifest "
                    "-- a rule that moved after the freeze is not the rule the verdict is read "
                    "against"
                )
    configuration = manifest.get("configuration", {})
    live = prereg(draws=configuration.get("draws", 1))["configuration"]
    for key in PINNED_CONFIGURATION:
        if configuration.get(key) != live[key]:
            complaints.append(
                f"configuration: {key} is {live[key]!r} here and "
                f"{configuration.get(key)!r} in the manifest"
            )
    if cohort not in COHORTS:
        complaints.append(f"cohorts: {cohort!r} is not one of {list(COHORTS)}")
        return complaints

    declared = {
        name: {int(pair[0]) for pair in pairs}
        for name, pairs in manifest.get("cohorts", {}).items()
    }
    mine = {int(data) for data, _ in seeds}
    for other in COHORTS:
        if other == cohort:
            continue
        overlap = sorted(mine & declared.get(other, set()))
        if overlap:
            complaints.append(
                f"cohorts: {len(overlap)} draw(s) of the {cohort} run are {other} draws -- an "
                f"effect would be reproduced on the draws that produced it, first at data seed "
                f"{overlap[0]}"
            )
    unknown = sorted(mine - declared.get(cohort, set()))
    if unknown:
        complaints.append(
            f"cohorts: {len(unknown)} draw(s) are not in the committed {cohort} cohort, first "
            f"at data seed {unknown[0]}"
        )
    expected = len(declared.get(cohort, set())) if draws_declared is None else draws_declared
    if expected and len(mine) < math.ceil(COMPLETENESS_FRACTION * expected):
        complaints.append(
            f"cohorts: {len(mine)} of {expected} declared {cohort} draws, below the "
            f"{COMPLETENESS_FRACTION:g} completeness clause -- the cell is unresolved by rule"
        )
    return complaints


# ------------------------------------------------------------------ the report


FIT_HEADERS = ("cell", "n", "arm", "draws", "sqrt(n) R", "score fails", "rounds", "closing", "exit")
CONTRAST_HEADERS = (
    "cell",
    "n",
    "contrast",
    "factor",
    "column",
    "draws",
    "mean",
    "95% interval",
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
        exits = sorted({r.exit_reason for r in good if r.exit_reason})
        body.append(
            (
                cell,
                str(n),
                arm,
                f"{len(good)}/{len(group)}",
                "--" if remaining.size == 0 else f"{remaining.mean():.4f}",
                f"{sum(r.score_failures for r in good)}",
                "--" if not good else f"{np.mean([r.rounds for r in good]):.1f}",
                "--" if not good else f"{np.mean([r.closing for r in good]):.1f}",
                ",".join(exits) or "--",
            )
        )
    return format_table(FIT_HEADERS, body)


def _contrast_table(rows: Sequence[ContrastRow]) -> str:
    body = [
        (
            row.cell,
            str(row.n),
            row.contrast,
            row.factor,
            row.column,
            str(row.draws),
            f"{row.mean:+.4f}",
            f"[{row.lower:+.4f}, {row.upper:+.4f}]",
            row.verdict,
        )
        for row in rows
        if row.column in {"root_n_remaining", "score_failures"}
    ]
    return format_table(CONTRAST_HEADERS, body)


READING = """
Reading the numbers
-------------------

**This localizes or it does not, and it may not branch.** F4's acceptance clause is all of it and
not some of it: the paired effect has to reproduce on the disjoint **audit** cohort and at both
sizes in the affected regime; it has to reduce `sqrt(n) R_remaining` **or** the score-failure
rate without worsening the other; the recorded score and state identities and the exact-law tests
have to stay valid; and the concordance has to either cover the selected construction or a new
derivation has to close the gap. A contrast that moves one column here is a question for F7, not
an answer.

**A `moved` verdict on the selection cohort alone is nothing.** The two cohorts are disjoint sets
of simulation draws, and the second exists because an effect assessed on the draws that produced
it is not an effect that has been reproduced. Read the audit rows beside the selection rows or
read neither.

**`flat` and `unresolved` are different, and the second is not a weak first.** `flat` says the
paired interval lies inside the declared negligible margin -- a real statement that the factor
does not move the column. `unresolved` says the study cannot tell. Only the first is evidence.

**No coverage number is in this table and none belongs in it.** Final coverage is F8's and only
F8's; a coverage reading taken here would be a release number taken before the change it is
meant to be about.

**The bound witness is recorded and never selected on.** The two truncation designs are declared
in advance; conditioning a contrast on the realized post-fit bound-active label would be
selecting the comparison on its own outcome. A `bound-active` result carries no theorem-backed
claim -- section 7 scopes the guarantee to the inactive-bound regime and covers neither
truncation.
"""


# ------------------------------------------------------------------ the CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phase", choices=("prereg", "run", "truncation"), required=True)
    parser.add_argument("--prereg", type=Path, default=Path("evidence/f4-construction/prereg.json"))
    parser.add_argument("--cohort", choices=COHORTS, default="selection")
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--sizes", nargs="+", type=int, default=list(SIZES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument(
        "--draws", type=int, default=None, help="debug only; defaults to the frozen count"
    )
    parser.add_argument("--quadrature-points", type=int, default=QUADRATURE_POINTS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=COHORT_SEED)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/drtmle-construction"))
    return parser


def _emit(payload: str, label: str) -> None:
    """Print a result so it can be recovered from a job log, digest first.

    Actions artefacts are served from a host the environment these studies are dispatched from
    cannot reach, so the log is the return path and ``scripts/recover_construction.sh`` is what
    takes it back out.  The digest is what makes that a recovery rather than a transcription.
    """
    raw = payload.encode("utf-8")
    print(f"{label}-SHA256 {hashlib.sha256(raw).hexdigest()}")
    print(f"{label}-BYTES {len(raw)}")
    print(f"--- BEGIN {label}.gz.base64 ---")
    import base64

    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    for start in range(0, len(encoded), 76):
        print(encoded[start : start + 76])
    print(f"--- END {label}.gz.base64 ---")


def main() -> int:
    args = build_parser().parse_args()

    if args.phase == "prereg":
        manifest = prereg(seed=args.seed, cells=args.cells, sizes=args.sizes, draws=args.draws)
        path = write_prereg(manifest, args.out / "prereg.json")
        print(f"wrote {path}")
        for size, block in sorted(manifest["rule"]["sizing"]["sizes"].items()):
            print(f"  n={size}: {block['arithmetic']} -- powered for {block['powered']}")
        print("cohorts: " + ", ".join(f"{k}={len(v)}" for k, v in manifest["cohorts"].items()))
        _emit(path.read_text(encoding="utf-8"), "PREREG")
        return 0

    if args.phase == "truncation":
        rows = truncation_reading()
        print(_truncation_table(rows))
        _emit("".join(json.dumps(asdict(row)) + "\n" for row in rows), "TRUNCATION")
        return 0

    if not args.prereg.exists():
        print(f"error: no committed preregistration at {args.prereg}", file=sys.stderr)
        print("Run --phase prereg, commit the manifest, then dispatch.", file=sys.stderr)
        return 2
    manifest = json.loads(args.prereg.read_text(encoding="utf-8"))

    # Each size takes its own committed prefix of the cohort, because the two sizes are sized
    # apart -- see SIZE_DRAWS. `--draws` narrows further and is a debug lever only.
    per_size = {
        n: (
            seeds_for(manifest, args.cohort, n)
            if args.draws is None
            else seeds_for(manifest, args.cohort, n)[: args.draws]
        )
        for n in args.sizes
    }
    complaints: list[str] = []
    for n, seeds in per_size.items():
        complaints.extend(
            validate_prereg(
                manifest,
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
            data_seed=int(data),
            fold_seed=int(fold),
            arms=tuple(a for a in ARMS if a in set(args.arms)),
            quadrature_points=args.quadrature_points,
        )
        for cell in args.cells
        for n in args.sizes
        for data, fold in per_size[n]
    ]
    fits = sum(len(p.arms) for p in payloads)
    print(
        f"F4 construction contrasts: {len(payloads)} draws, {fits} fits, cohort={args.cohort}, "
        + ", ".join(f"n={n}: {len(s)} draws" for n, s in sorted(per_size.items()))
    )
    produced = Parallel(n_jobs=args.jobs)(delayed(one_draw)(p) for p in payloads)
    rows = [row for group in produced or [] for row in group]

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    fits_path = args.out / f"{stamp}.jsonl"
    contrasts = contrast_rows(rows, seed=args.seed)
    contrasts_path = args.out / f"{stamp}-contrasts.jsonl"
    fits_path.write_text("".join(json.dumps(asdict(row)) + "\n" for row in rows), encoding="utf-8")
    contrasts_path.write_text(
        "".join(json.dumps(asdict(row)) + "\n" for row in contrasts), encoding="utf-8"
    )

    print()
    print(_fit_table(rows))
    print()
    print(_contrast_table(contrasts))
    print(READING)
    _emit(contrasts_path.read_text(encoding="utf-8"), "CONTRASTS")

    errored = sum(1 for row in rows if row.error)
    if errored:
        print(f"{errored} fit row(s) recorded an error", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
