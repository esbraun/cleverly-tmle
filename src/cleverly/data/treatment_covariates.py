r"""Treatment-mechanism covariates: a subset of the adjustment set that the mechanism reads.

A design can declare that the treatment mechanism conditions on a subset :math:`W_g` of the
adjustment covariates :math:`W`, while the outcome regression and every other nuisance keep
the full set.  The declaration restricts the observed-data model.  It does not change the
estimand.

The declaration lives on the **data** (:attr:`~cleverly.data.CausalData.treatment_covariate_names`),
as a declared known mechanism does, so every reader that subsets rows carries it.  This module
holds the two things that every layer shares: the check of a declaration, and the pinned text
that a design prints.

==========================  ===============================================================
reader                      covariates it reads
==========================  ===============================================================
treatment mechanism         :math:`W_g` (``CausalData.mechanism_covariates``)
treatment observation       :math:`W`
outcome observation         :math:`W`
outcome regression          :math:`W`
policy rule evaluation      :math:`W`
==========================  ===============================================================
"""

from __future__ import annotations

from collections.abc import Sequence

from ..exceptions import DataError

__all__ = [
    "DRTMLE_LINE_TEMPLATE",
    "POOL_LINE_TEMPLATE",
    "RESTRICTION_TEMPLATE",
    "check_treatment_covariates",
    "drtmle_restriction_line",
    "known_and_restricted",
    "pool_line",
    "treatment_mechanism_restriction",
]

#: The statistical-model restriction that a design with treatment-mechanism covariates prints.
#: ``{Wg}`` is the set of covariate names, written as ``{name, name}``.
RESTRICTION_TEMPLATE = (
    "statistical-model restriction, treatment-mechanism covariates {Wg}: the treatment "
    "mechanism depends on the adjustment covariates only through {Wg}. It restricts the "
    "observed-data model and does not change the estimand. The fit is consistent when the "
    "treatment mechanism fitted on {Wg} is consistent, which needs the restriction, or when "
    "the outcome regression is consistent. The interval needs the restriction and both "
    "nuisances consistent at the usual product rate. When the restriction fails, the "
    "remainder is first order in the outcome-regression error. To exclude instruments, omit "
    "them from adjustment= instead: the outcome regression and the treatment mechanism then "
    "both use {Wg}."
)


#: The result-summary line of a DR-TMLE fit with treatment-mechanism covariates.  It points to
#: the one-wrong table of the DR-TMLE theorem page, whose mechanism-wrong rows decide the
#: interval when the restriction fails.
DRTMLE_LINE_TEMPLATE = (
    "DRTMLE with treatment-mechanism covariates {Wg}: when the restriction fails, the "
    "treatment mechanism is inconsistent, and the mechanism-wrong rows of the one-wrong "
    "table in docs/technical-reference/dr-tmle/theorem.md decide the interval."
)

#: The line a collaborative selection summary prints when its pool is a subset of the
#: adjustment set.
POOL_LINE_TEMPLATE = "collaborative candidate pool: {Wg}"


def _braced(names: Sequence[str]) -> str:
    return "{" + ", ".join(names) + "}"


def drtmle_restriction_line(names: Sequence[str]) -> str:
    """Return the pinned DR-TMLE summary line for a set of treatment-mechanism covariates.

    Parameters
    ----------
    names : sequence of str
        The covariates the treatment mechanism reads, as the caller named them.

    Returns
    -------
    str
        :data:`DRTMLE_LINE_TEMPLATE` with ``{Wg}`` written as ``{name, name}``.
    """
    return DRTMLE_LINE_TEMPLATE.replace("{Wg}", _braced(names))


def pool_line(names: Sequence[str]) -> str:
    """Return the selection-summary line that names a restricted candidate pool.

    Parameters
    ----------
    names : sequence of str
        The covariates the candidate path draws from.

    Returns
    -------
    str
        :data:`POOL_LINE_TEMPLATE` with ``{Wg}`` written as ``{name, name}``.
    """
    return POOL_LINE_TEMPLATE.replace("{Wg}", _braced(names))


def treatment_mechanism_restriction(names: Sequence[str]) -> str:
    """Return the pinned restriction text for a set of treatment-mechanism covariates.

    Parameters
    ----------
    names : sequence of str
        The covariates the treatment mechanism reads, as the caller named them.

    Returns
    -------
    str
        :data:`RESTRICTION_TEMPLATE` with ``{Wg}`` written as ``{name, name}``.
    """
    return RESTRICTION_TEMPLATE.replace("{Wg}", _braced(names))


def check_treatment_covariates(
    names: Sequence[str] | str,
    adjustment: Sequence[str],
    *,
    argument: str,
    base: str,
) -> tuple[str, ...]:
    """Check a declared subset of the adjustment set.

    Parameters
    ----------
    names : sequence of str
        The declared treatment-mechanism covariates.  An empty sequence is allowed.
    adjustment : sequence of str
        The adjustment covariates the declaration must be a subset of.
    argument : str
        The declaration's own name, for the message.
    base : str
        The adjustment argument's own name, for the message.

    Returns
    -------
    tuple of str
        The names, in the order given.

    Raises
    ------
    DataError
        If a name repeats (R-h6) or lies outside ``adjustment`` (R-h1).
    """
    if isinstance(names, str):
        raise DataError(f"{argument} takes a sequence of column names; got the string {names!r}")
    declared = tuple(names)
    repeated = sorted({name for name in declared if declared.count(name) > 1})
    if repeated:
        raise DataError(f"{argument} repeats {repeated}")
    outside = [name for name in declared if name not in set(adjustment)]
    if outside:
        raise DataError(
            f"{argument} names {outside} outside {base}=. The treatment mechanism conditions "
            "on a subset of the adjustment set"
        )
    return declared


def known_and_restricted(argument: str) -> DataError:
    """Build the refusal of a restriction beside a known treatment mechanism (R-h4).

    Parameters
    ----------
    argument : str
        The restriction's own name, for the message.

    Returns
    -------
    DataError
        The refusal, for the caller to raise.
    """
    return DataError(
        f"{argument}= restricts a fitted treatment mechanism, and treatment_probabilities= "
        "supplies a known one. Remove one of them"
    )
