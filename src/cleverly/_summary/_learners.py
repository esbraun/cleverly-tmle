"""Plain names for the learners a fit used, and whether each is parametric."""

from __future__ import annotations

from typing import Any

from . import vocabulary


def _steps(learner: Any) -> list[Any] | None:
    """The steps of a scikit-learn ``Pipeline``, or ``None`` for any other object."""
    steps = getattr(learner, "steps", None)
    if type(learner).__name__ != "Pipeline" or not isinstance(steps, list):
        return None
    return [step for _, step in steps]


def _entries(learner: Any) -> list[Any] | None:
    """The candidates of a ``SuperLearner``, or ``None`` when it uses the default library."""
    library = getattr(learner, "library", None)
    if library is None:
        return None
    return [entry[1] if isinstance(entry, tuple) else entry for entry in library]


def learner_name(learner: Any) -> str:
    """The plain name of one learner, as a summary prints it.

    ``None`` (the package default), a super learner (with its candidate count), a
    ``Pipeline`` (its final step, after its other steps) and the classes of
    :data:`~cleverly._summary.vocabulary.LEARNER_NAMES` get plain names. Any other class
    prints its own name, which is what the user wrote.

    Parameters
    ----------
    learner : object or None
        A learner specification as the estimator holds it.

    Returns
    -------
    str
        The name.
    """
    if learner is None:
        return vocabulary.DEFAULT_LEARNER
    kind = type(learner).__name__
    if kind == "SuperLearner":
        entries = _entries(learner)
        if entries is None:
            return vocabulary.DEFAULT_LEARNER
        return vocabulary.SUPER_LEARNER.format(k=len(entries))
    steps = _steps(learner)
    if steps is not None:
        named = [step for step in steps if step is not None and step != "passthrough"]
        if not named:
            return "passthrough"
        final = learner_name(named[-1])
        before = [learner_name(step) for step in named[:-1]]
        return final if not before else f"{final} after {' and '.join(before)}"
    return vocabulary.LEARNER_NAMES.get(kind, kind)


def is_parametric(learner: Any) -> bool:
    """Whether a learner is a fixed-dimension parametric model.

    A class in :data:`~cleverly._summary.vocabulary.PARAMETRIC_LEARNERS`, matched by exact
    class name, or a ``Pipeline`` whose final step is one and whose other steps only
    rescale. Everything else counts as flexible: the default super learner, any other
    ensemble, lasso-type learners, and any class this list does not name.

    Parameters
    ----------
    learner : object or None
        A learner specification.

    Returns
    -------
    bool
        ``True`` for a parametric model.
    """
    if learner is None:
        return False
    steps = _steps(learner)
    if steps is not None:
        if not steps:
            return False
        *before, final = steps
        return is_parametric(final) and all(
            step is None
            or step == "passthrough"
            or type(step).__name__ in vocabulary.PARAMETRIC_STEPS
            for step in before
        )
    if type(learner).__name__ not in vocabulary.PARAMETRIC_LEARNERS:
        return False
    # An l1 or elastic-net penalty selects covariates from the data, as a lasso does.
    penalty = getattr(learner, "penalty", None)
    l1_ratio = getattr(learner, "l1_ratio", None)
    if penalty in ("l1", "elasticnet"):
        return False
    return not (penalty not in ("l2", None, "none") and l1_ratio not in (None, 0, 0.0))
