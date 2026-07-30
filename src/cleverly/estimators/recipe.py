"""A JSON-safe snapshot of how an estimator was configured.

:class:`~cleverly.TMLEResult` holds a live reference to the :class:`~cleverly.TMLE`
that produced it, which is what makes ``retarget``-based sensitivity analysis work
and what makes the result unserialisable.  Splitting the two is the point of this
module: the *recipe* is the settings, always serialisable; the *estimator* is the
live object, needed only for a genuine refit.

Most of the constructor is already JSON-safe -- scalars, strings, tuples.  The four
learner slots are not, because they accept an arbitrary scikit-learn estimator.  The
recipe stores those when they are a library *specification* (``"glm"``, ``"default"``,
a list of names) and records :attr:`TMLERecipe.learners_reconstructible` as ``False``
when they are not.  That flag is the honest boundary, and it is checked rather than
hoped for: a round-tripped result can do everything that goes through ``retarget``,
and raises a specific error for the two analyses that genuinely refit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

__all__ = ["LEARNER_SLOTS", "SETTING_NAMES", "TMLERecipe"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tmle import TMLE

#: Constructor arguments holding a learner or a learner specification.
LEARNER_SLOTS: tuple[str, ...] = (
    "outcome_learner",
    "treatment_learner",
    "missingness_learner",
    "intermediate_learner",
)

#: Every other constructor argument.  All are scalars, strings or ``None``.
SETTING_NAMES: tuple[str, ...] = (
    "family",
    "fluctuation",
    "targeting",
    "cross_fit",
    "targeting_scheme",
    "cv_evaluation",
    "n_folds",
    "learner_folds",
    "g_bounds",
    "q_bounds",
    "alpha",
    "nuisance_bound",
    "target_weights",
    "screen_treatment",
    "screen_threshold",
    "min_retain",
    "estimands",
    "alpha_sig",
    "n_bootstrap",
    "bootstrap_resampling",
    "simultaneous",
    "n_multiplier",
    "multiplier_kind",
    "step_size",
    "max_iter",
    "tol",
    "random_state",
    "run_id",
    "n_jobs",
)

#: Constructor arguments handled by hand rather than by :func:`_jsonable`, so that
#: :func:`_subclass_settings` does not pick them up and try.
_HANDLED_ELSEWHERE: frozenset[str] = frozenset({"interventions"})


def _is_specification(value: Any) -> bool:
    """Is this a library name or list of names, rather than a fitted estimator?"""
    if value is None or isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, str) for item in value)
    return False


@dataclass(frozen=True)
class TMLERecipe:
    """The settings of a fit, without the objects.

    Attributes
    ----------
    learners_reconstructible:
        ``True`` when every learner slot held a library specification, so the
        estimator can be rebuilt exactly.  ``False`` when at least one held a
        scikit-learn estimator, which cannot be described by a string -- the
        settings are still recorded, but rebuilding would silently substitute the
        default library, and that is refused instead.
    """

    settings: dict[str, Any]
    learners: dict[str, Any]
    #: The declared regimes, as ``{"level": ..., "name": ...}`` -- but only when every
    #: one of them is :class:`~cleverly.interventions.Static`.  A ``Rule`` or a
    #: ``Stochastic`` regime holds a *callable*, which no recipe can describe, so such a
    #: fit is recorded as unreconstructible on the same terms as a fitted learner: the
    #: numbers a result already carries are unaffected, and only the analyses that refit
    #: need the estimator back.
    interventions: list[dict[str, Any]] = field(default_factory=list)
    learners_reconstructible: bool = True
    unreconstructible_slots: tuple[str, ...] = ()
    class_name: str = "TMLE"
    class_module: str = "cleverly.estimators.tmle"
    #: Constructor arguments of a subclass that the base recipe does not know about.
    extra_settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_estimator(cls, estimator: TMLE) -> TMLERecipe:
        settings = {
            name: _jsonable(getattr(estimator, name))
            for name in SETTING_NAMES
            if hasattr(estimator, name)
        }
        learners: dict[str, Any] = {}
        unreconstructible: list[str] = []
        for slot in LEARNER_SLOTS:
            value = getattr(estimator, slot, None)
            if _is_specification(value):
                learners[slot] = _jsonable(value)
            else:
                unreconstructible.append(slot)
                learners[slot] = None
        interventions = _interventions_to(getattr(estimator, "interventions", ()))
        if interventions is None:
            unreconstructible.append("interventions")
        return cls(
            settings=settings,
            learners=learners,
            interventions=interventions or [],
            learners_reconstructible=not unreconstructible,
            unreconstructible_slots=tuple(unreconstructible),
            class_name=type(estimator).__name__,
            class_module=type(estimator).__module__,
            extra_settings=_subclass_settings(estimator),
        )

    def build(self) -> TMLE:
        """Rebuild the estimator.

        Raises when a learner slot held a fitted object rather than a library
        specification: substituting the default library would produce an estimator
        that looks right and is not the one that made these numbers.
        """
        if not self.learners_reconstructible:
            slots = ", ".join(self.unreconstructible_slots)
            held = (
                "a rule or a stochastic density, which is a callable"
                if self.unreconstructible_slots == ("interventions",)
                else "a scikit-learn estimator rather than a library name"
            )
            raise ValueError(
                f"this result's {slots} held {held}, so the estimator cannot be rebuilt "
                "from the recipe. Everything reached through retarget() -- positivity, "
                "truncation curves, the score check, the bootstrap -- works without it, "
                "because the evaluated regime densities are stored with the nuisances; "
                "only refit-based analyses (refute, benchmark) need the original "
                "estimator object."
            )
        module = __import__(self.class_module, fromlist=[self.class_name])
        klass = getattr(module, self.class_name)
        kwargs = {**self.settings, **self.extra_settings}
        kwargs = {key: _restore(key, value) for key, value in kwargs.items()}
        kwargs.update({slot: self.learners[slot] for slot in LEARNER_SLOTS})
        if self.interventions:
            kwargs["interventions"] = _interventions_from(self.interventions)
        return klass(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "learners": self.learners,
            "interventions": self.interventions,
            "learners_reconstructible": self.learners_reconstructible,
            "unreconstructible_slots": list(self.unreconstructible_slots),
            "class_name": self.class_name,
            "class_module": self.class_module,
            "extra_settings": self.extra_settings,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TMLERecipe:
        return cls(
            settings=payload["settings"],
            learners=payload["learners"],
            interventions=payload.get("interventions", []),
            learners_reconstructible=payload["learners_reconstructible"],
            unreconstructible_slots=tuple(payload.get("unreconstructible_slots", ())),
            class_name=payload.get("class_name", "TMLE"),
            class_module=payload.get("class_module", "cleverly.estimators.tmle"),
            extra_settings=payload.get("extra_settings", {}),
        )


#: Settings that must come back as a tuple rather than the list JSON gives.
_TUPLE_SETTINGS = frozenset({"g_bounds", "q_bounds"})


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _restore(key: str, value: Any) -> Any:
    if key in _TUPLE_SETTINGS and isinstance(value, list):
        return tuple(value)
    return value


def _interventions_to(interventions: Any) -> list[dict[str, Any]] | None:
    """The declared regimes as JSON, or ``None`` when one of them holds a callable."""
    from ..interventions import Static

    recorded: list[dict[str, Any]] = []
    for intervention in interventions or ():
        if not isinstance(intervention, Static):
            return None
        recorded.append({"level": _jsonable(intervention.level), "name": intervention.name})
    return recorded


def _interventions_from(recorded: list[dict[str, Any]]) -> tuple[Any, ...]:
    from ..interventions import Static

    return tuple(Static(item["level"], name=item["name"]) for item in recorded)


def _subclass_settings(estimator: TMLE) -> dict[str, Any]:
    """Constructor arguments a subclass added, found by inspecting its signature."""
    import inspect

    known = set(SETTING_NAMES) | set(LEARNER_SLOTS) | _HANDLED_ELSEWHERE
    try:
        parameters = inspect.signature(type(estimator).__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        return {}
    return {
        name: _jsonable(getattr(estimator, name))
        for name in parameters
        if name not in known and name != "self" and hasattr(estimator, name)
    }
