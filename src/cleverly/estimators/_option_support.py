"""The support conditions of ``patience=``, the one targeting option no study row keys.

A fit that declares ``patience=`` runs, and its estimates carry the experimental label
(:func:`patience_conditions`).  X76 part (c) in ``docs/roadmap.md`` tracks the greedy path's
study; on a pre-ordered path the condition is permanent.

The other options have no condition.  The support key records each of them, so a fit that
declares one lands on the row of the study that measured it or matches no row and reads
experimental:

=================================  =================================================  ==========
option                             registered rows                                    study
=================================  =================================================  ==========
``stopping=``                      ``T52`` (point), ``G55`` (``LTMLE``)               X76 (a)
``outcome_by_arm=True``            ``T53``                                            X76 (b)
``treatment_adjustment=``          ``T54``                                            X76 (d)
=================================  =================================================  ==========

Under a declared rule the key also records ``step_bound=`` and ``iterate=``, which no row
declares, so such a fit reads experimental.  On the effect-modification axis the key keeps a
pooled projection fit off ``T49`` and ``T50``, whose study fits one outcome regression per arm.

``iterate="best"`` and ``step_bound=`` keep the support of the fit under
``stopping="relative"``.  The walk then stops only at a fit that solves the score equation to
``tol``.  The submodel loss is concave in the fluctuation coefficient, so that fit is the one the
Newton path reaches, to ``tol``.  A walk that stops short fails ``score_check()``, as a Newton
solve that stops short does.

Each function returns :class:`~cleverly._refusal.Refusal` conditions for a
:class:`~cleverly._support.Preflight`.
"""

from __future__ import annotations

from typing import Any

from .._refusal import Refusal, RefusalReason

__all__ = [
    "OPTION_STUDIES_ROADMAP",
    "patience_conditions",
    "point_option_conditions",
]

#: The roadmap item that tracks the registered studies of these options.
OPTION_STUDIES_ROADMAP = "X76"


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
        The ``patience=`` condition when the fit declares it, else none.
    """
    return patience_conditions(estimator)
