"""The support conditions of the targeting and fitting options.

A fit that declares one of these options runs, and its estimates carry the experimental label.
Each option is implemented and tested on exact laws, and no registered study measures a fit that
uses it. The registered studies ran each estimator at the defaults below, so their evidence does
not reach a fit that leaves them:

========================  ==================  =====================================================
option                    default             why the evidence of the default does not cover it
========================  ==================  =====================================================
``stopping=``             ``"relative"``      the rule leaves a score residual, which the
                                              interval adds, and no registered study measures
                                              the widened interval
``patience=``             ``None``            another collaborative selection; on a pre-ordered
                                              path it can stop before a needed candidate at
                                              every sample size (permanent)
========================  ==================  =====================================================

``outcome_by_arm=`` and ``treatment_adjustment=`` have no condition.  The support key records
both, so a fit that declares one lands on its registered row (``T53`` and ``T54``, studies
``option-outcome-by-arm`` and ``option-treatment-adjustment``) or matches no row and reads
experimental.  On the effect-modification axis the key also keeps a pooled projection fit off
``T49`` and ``T50``, whose study fits one outcome regression per arm.

``iterate="best"`` and ``step_bound=`` keep the support of the fit under
``stopping="relative"``.  The walk then stops only at a fit that solves the score equation to
``tol``.  The submodel loss is concave in the fluctuation coefficient, so that fit is the one the
Newton path reaches, to ``tol``.  A walk that stops short fails ``score_check()``, as a Newton
solve that stops short does.

Each function returns :class:`~cleverly._refusal.Refusal` conditions for a
:class:`~cleverly._support.Preflight`.  X76 in ``docs/roadmap.md`` tracks the studies.
"""

from __future__ import annotations

from typing import Any

from .._refusal import Refusal, RefusalReason

__all__ = [
    "OPTION_STUDIES_ROADMAP",
    "patience_conditions",
    "point_option_conditions",
    "stopping_conditions",
]

#: The roadmap item that tracks the registered studies of these options.
OPTION_STUDIES_ROADMAP = "X76"


def stopping_conditions(stopping: str) -> list[Refusal]:
    """The condition of a declared stopping rule, or none under ``"relative"``.

    Parameters
    ----------
    stopping : str
        The declared stopping rule.

    Returns
    -------
    list of Refusal
        One condition under ``"scaled_var"`` or ``"sample_size"``, else none.
    """
    if stopping == "relative":
        return []
    return [
        Refusal(
            RefusalReason.INFERENCE_GAP,
            f"The declared rule stopping={stopping!r} stops adjusting the outcome predictions "
            "before the estimating equation is solved. The interval adds the remainder it "
            "leaves to its half-width, and no registered simulation study measures that "
            "interval yet.",
            alternative=(
                "stopping='relative' solves the score equation to tol, as the registered "
                "studies do."
            ),
            roadmap=OPTION_STUDIES_ROADMAP,
        )
    ]


def patience_conditions(estimator: Any) -> list[Refusal]:
    """The condition of ``patience=`` on a C-TMLE path, or none for the whole path.

    On a pre-ordered path (``"ordered"``, ``"discrete"``, ``"lasso"``, ``"learners"``) the
    order is fixed before any risk is read, so a candidate the outcome regression needs can sit
    after ``patience`` candidates that only add noise.  The counter then stops before it with a
    probability that does not shrink with the sample size, and the estimate keeps a bias no
    study can rule out.  That condition is permanent, so it names no roadmap item.  The greedy
    path adds the candidate that lowers the risk most at each step; its condition names the
    study that measures it.

    Parameters
    ----------
    estimator : CTMLE
        The estimator, as constructed.  It is read and never modified.

    Returns
    -------
    list of Refusal
        One condition when ``patience`` is set, else none.
    """
    patience = getattr(estimator, "patience", None)
    if patience is None:
        return []
    if str(getattr(estimator, "strategy", "greedy")) == "greedy":
        return [
            Refusal(
                RefusalReason.INFERENCE_GAP,
                f"The setting patience={patience} can stop the cross-validated path before "
                "its best candidate, and no registered study covers that selection.",
                alternative="patience=None scores the whole path, as the registered studies do.",
                roadmap=OPTION_STUDIES_ROADMAP,
            )
        ]
    return [
        Refusal(
            RefusalReason.INFERENCE_GAP,
            f"The setting patience={patience} stops a pre-ordered candidate path after "
            f"{patience} candidates whose cross-validated risk is not a new minimum. A "
            "candidate the outcome regression needs that is ordered after them is never "
            "scored, so the estimate can stay biased at every sample size.",
            alternative=(
                "patience=None scores the whole path, so a candidate ordered late is still scored."
            ),
            roadmap=None,
        )
    ]


def point_option_conditions(estimator: Any, data: Any) -> list[Refusal]:
    """The conditions of the point-treatment options a fit declares.

    Parameters
    ----------
    estimator : TMLE
        The estimator, as constructed.  It is read and never modified.
    data : CausalData
        The prepared data.  No remaining condition reads it; the support key reads its
        treatment-mechanism covariates.

    Returns
    -------
    list of Refusal
        One condition per declared option, in the order of the module table.
    """
    conditions = stopping_conditions(str(getattr(estimator, "stopping", "relative")))
    conditions += patience_conditions(estimator)
    return conditions
