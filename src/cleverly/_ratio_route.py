"""How a fit estimates a modified treatment policy's density ratio: the resolved ``ratio=``.

``ratio=None``, the default of every estimator that takes it, is resolved once per fit by
:func:`resolve_ratio`, before any check reads the value.  The classifier route
(Díaz, Williams, Hoffman and Schenck 2023, Section 5.4) serves a policy on a continuous or
vector dose.  The binned density serves every other fit, and these even with such a dose:
a fit that sets ``density_bins``, a treatment learner that declares ``bin_design``, and an
estimator or composition that refuses the classifier.  An explicit ``"density"`` or
``"classifier"`` keeps its meaning and its refusals.  The resolved route and the rule that
chose it are recorded on the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._refusal import Refusal, RefusalReason
from .exceptions import MethodConfigurationError

__all__ = [
    "ROUTES",
    "RatioRoute",
    "check_ratio",
    "refuse_classifier_settings",
    "resolve_ratio",
    "warn_isolated_clusters",
]

#: The two routes ``ratio=`` names.
ROUTES = ("density", "classifier")

#: The rules that choose a route, as the result records them.
REQUESTED = "requested"
POLICY_DOSE = "default: a policy on a continuous or vector dose"
DENSITY_BINS = "default: density_bins is set"
BIN_DESIGN = "default: the treatment learner declares bin_design"
NO_DOSE = "default: no policy ratio on a continuous or vector dose"


@dataclass(frozen=True)
class RatioRoute:
    """A fit's policy-ratio route and the rule that chose it.

    Parameters
    ----------
    route : str
        ``"density"`` or ``"classifier"``.
    reason : str
        The rule: the request itself, or the default rule that applied.
    """

    route: str
    reason: str

    def describe(self) -> str:
        """One line for a summary: the route, and the rule that chose it."""
        if self.reason == REQUESTED:
            return f"{self.route} (requested)"
        return f"{self.route} ({self.reason})"


def check_ratio(ratio: Any) -> None:
    """Refuse a ``ratio=`` that names no route.

    Parameters
    ----------
    ratio : object
        The setting.

    Raises
    ------
    ValueError
        If it is not ``None``, ``"density"`` or ``"classifier"``.
    """
    if ratio is not None and ratio not in ROUTES:
        raise ValueError(
            f"ratio must be None, 'density' or 'classifier'; got {ratio!r}. 'density' "
            "estimates a policy's ratio from a binned conditional density of the dose, "
            "'classifier' from a stacked classification (Diaz et al. 2023, Section 5.4), and "
            "None chooses per fit: the classifier for a policy on a continuous or vector dose, "
            "the density otherwise"
        )


def _declares_bin_design(learner: Any) -> bool:
    return getattr(learner, "bin_design", None) is not None


def refuse_classifier_settings(ratio: Any, density_bins: Any, learner: Any) -> None:
    """Refuse ``ratio="classifier"`` beside a setting only the density route reads.

    Parameters
    ----------
    ratio : object
        The requested route.
    density_bins : object
        The ``density_bins`` setting.
    learner : object
        The treatment learner.

    Raises
    ------
    MethodConfigurationError
        If ``ratio="classifier"`` is combined with ``density_bins`` or a learner that
        declares ``bin_design``.
    """
    if ratio != "classifier":
        return
    if density_bins is not None:
        setting = f"density_bins={density_bins!r}"
    elif _declares_bin_design(learner):
        setting = f"a treatment learner that declares bin_design={learner.bin_design!r}"
    else:
        return
    raise MethodConfigurationError.from_refusal(
        Refusal(
            RefusalReason.INVALID_DECLARATION,
            f"ratio='classifier' estimates the policy ratio by a stacked classification and "
            f"fits no binned density, so {setting} would change nothing.",
            alternative="Drop the density setting, or pass ratio='density' (or leave ratio "
            "unset, which then resolves to the density route).",
        )
    )


def resolve_ratio(
    ratio: Any,
    *,
    policy_dose: bool,
    density_bins: Any,
    learner: Any,
    density_only: str | None = None,
) -> RatioRoute:
    """The route a fit estimates its policy ratio by, and the rule that chose it.

    Parameters
    ----------
    ratio : object
        The requested ``ratio=``: ``None`` or a route.
    policy_dose : bool
        Whether the fit estimates a policy's ratio on a continuous or vector dose.
    density_bins : object
        The ``density_bins`` setting.
    learner : object
        The treatment learner.
    density_only : str or None
        Why this fit reads the density route only, as the result records it, or ``None``.

    Returns
    -------
    RatioRoute
        The route and its rule.
    """
    check_ratio(ratio)
    if ratio is not None:
        refuse_classifier_settings(ratio, density_bins, learner)
        return RatioRoute(str(ratio), REQUESTED)
    if density_only is not None:
        return RatioRoute("density", f"default: {density_only}")
    if not policy_dose:
        return RatioRoute("density", NO_DOSE)
    if density_bins is not None:
        return RatioRoute("density", DENSITY_BINS)
    if _declares_bin_design(learner):
        return RatioRoute("density", BIN_DESIGN)
    return RatioRoute("classifier", POLICY_DOSE)


def warn_isolated_clusters(doses: Mapping[str, Any]) -> None:
    """Warn of each isolated dose cluster on a fit whose policy ratio is the classifier's.

    Parameters
    ----------
    doses : mapping of str to array-like
        Each dose's observed values, by the name the warning gives it.  At a longitudinal
        node these are the doses of the rows at risk.
    """
    from ._warnings import warn
    from .exceptions import DataWarning
    from .learners.density import isolated_dose_clusters

    for name, values in doses.items():
        for cluster in isolated_dose_clusters(np.asarray(values, dtype=float)):
            warn(
                f"{name} has {cluster.rows} observed doses between {cluster.lower:.4g} and "
                f"{cluster.upper:.4g}, beyond an empty stretch of the dose. The classifier "
                "estimate of the policy ratio can smooth over such a cluster; "
                "ratio='density' fits a binned density whose grid keeps it "
                "(cleverly.learners.isolated_dose_clusters).",
                DataWarning,
                stacklevel=3,
            )
