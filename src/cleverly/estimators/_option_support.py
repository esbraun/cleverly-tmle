"""The support conditions of the targeting and fitting options.

A fit that declares one of these options runs, and its estimates carry the experimental label.
Each option is implemented and tested on exact laws, and no registered study measures a fit that
uses it. The registered studies ran each estimator at the defaults below, so their evidence does
not reach a fit that leaves them:

========================  ==================  =====================================================
option                    default             why the evidence of the default does not cover it
========================  ==================  =====================================================
``stopping=``             ``"relative"``      the rule leaves a score residual by design, and the
                                              interval coverage depends on it
``outcome_by_arm=``       ``False``           another outcome regression
``patience=``             ``None``            another collaborative selection
``treatment_adjustment=`` the adjustment set  another treatment mechanism, under a restriction
========================  ==================  =====================================================

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

__all__ = ["OPTION_STUDIES_ROADMAP", "point_option_conditions", "stopping_conditions"]

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
            f"The declared rule stopping={stopping!r} ends the targeting at a score residual "
            "that the rule allows, and no registered study measures the interval at that "
            "residual.",
            alternative=(
                "stopping='relative' solves the score equation to tol, as the registered "
                "studies do. The summary states the coverage bound at the residual."
            ),
            roadmap=OPTION_STUDIES_ROADMAP,
        )
    ]


def point_option_conditions(estimator: Any, data: Any) -> list[Refusal]:
    """The conditions of the point-treatment options a fit declares.

    Parameters
    ----------
    estimator : TMLE
        The estimator, as constructed.  It is read and never modified.
    data : CausalData
        The prepared data, which carries the treatment-mechanism covariates.

    Returns
    -------
    list of Refusal
        One condition per declared option, in the order of the module table.
    """
    conditions = stopping_conditions(str(getattr(estimator, "stopping", "relative")))
    if getattr(estimator, "outcome_by_arm", False):
        conditions.append(
            Refusal(
                RefusalReason.INFERENCE_GAP,
                "The setting outcome_by_arm=True fits one outcome regression per arm, and no "
                "registered study covers that outcome fit.",
                alternative=(
                    "outcome_by_arm=False fits the one outcome regression the registered "
                    "studies use."
                ),
                roadmap=OPTION_STUDIES_ROADMAP,
            )
        )
    patience = getattr(estimator, "patience", None)
    if patience is not None:
        conditions.append(
            Refusal(
                RefusalReason.INFERENCE_GAP,
                f"The setting patience={patience} can stop the cross-validated path before "
                "its best candidate, and no registered study covers that selection.",
                alternative="patience=None scores the whole path, as the registered studies do.",
                roadmap=OPTION_STUDIES_ROADMAP,
            )
        )
    if getattr(data, "treatment_covariate_names", None) is not None:
        conditions.append(
            Refusal(
                RefusalReason.INFERENCE_GAP,
                "The setting treatment_adjustment= fits the treatment mechanism on a subset of "
                "the adjustment set, and no registered study covers that restriction.",
                alternative=(
                    "Omit treatment_adjustment= (treatment_covariates= in TMLE.fit) to fit the "
                    "mechanism on the whole adjustment set."
                ),
                roadmap=OPTION_STUDIES_ROADMAP,
            )
        )
    return conditions
