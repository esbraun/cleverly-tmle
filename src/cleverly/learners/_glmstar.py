"""The optional glmstar accelerator of the L1-penalized logistic path.

``CTMLE(l1_solver="glmstar")`` starts each positive penalty of a path below the null gradient
from glmstar's solution of the same objective (R ``glmnet``'s compiled core, through the
``glmstar`` package), and :mod:`._l1_logistic` checks and polishes it to the built-in
Karush-Kuhn-Tucker tolerance. This module loads glmstar, maps the standardized problem onto
it, and returns the starting points. It never accepts a solution itself.

glmstar installs the top-level module ``glmnet``, which the unrelated ``python-glmnet``
package also uses, so the loader checks that the imported module belongs to the ``glmstar``
distribution. glmstar is GPL-2.0-only and is installed with ``pip install
"cleverly[glmstar]"``; it is imported only when a fit asks for it.
"""

from __future__ import annotations

import functools
import importlib
import importlib.metadata
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .._typing import FloatArray

if TYPE_CHECKING:
    from ._l1_logistic import _Problem

#: The oldest glmstar release the adapter was checked against.
MINIMUM_VERSION = (0, 7, 1)

_INSTALL = 'pip install "cleverly[glmstar]"'


class GlmstarFallback(Exception):
    """A declared numerical outcome after which the path is solved by the built-in solver.

    Parameters
    ----------
    reason : str
        ``"nonconvergence"``, ``"fatal"``, ``"short_path"``, ``"nonfinite"``, ``"polish"``
        or ``"absent_class"``.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split(".")[:3]:
        digits = re.match(r"\d*", piece)
        parts.append(int(digits.group()) if digits and digits.group() else 0)
    return tuple(parts)


def _below_floor(text: str) -> bool:
    """Whether a release is older than :data:`MINIMUM_VERSION`; a pre-release or development
    build of the floor itself (``0.7.1rc1``, ``0.7.1.dev0``) precedes it."""
    release = _version_tuple(text)
    if release != MINIMUM_VERSION:
        return release < MINIMUM_VERSION
    return re.search(r"(a|b|rc|dev)\d*", text.lower()) is not None


def _jerr(estimator: Any) -> int:
    """glmstar's error code after a fit: negative non-fatal, positive fatal, 0 none."""
    fit = getattr(estimator, "_fit", None)
    try:
        return int(fit["jerr"]) if fit is not None else 0
    except (KeyError, TypeError, ValueError):
        return 0


@functools.cache
def load() -> tuple[Any, Any, Any]:
    """Import glmstar and return its binary path, multinomial path and control classes.

    Returns
    -------
    tuple
        ``(LogNet, MultiClassNet, FastNetControl)``.

    Raises
    ------
    ImportError
        When glmstar is not installed, is older than :data:`MINIMUM_VERSION`, or the
        importable ``glmnet`` module belongs to another distribution.
    """
    try:
        version = importlib.metadata.version("glmstar")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            "l1_solver='glmstar' needs the glmstar package, which is not installed. "
            f"Install it with {_INSTALL}, or use l1_solver='builtin'."
        ) from None
    if _below_floor(version):
        floor = ".".join(str(v) for v in MINIMUM_VERSION)
        raise ImportError(
            f"l1_solver='glmstar' needs glmstar {floor} or later; {version} is installed. "
            f"Upgrade it with {_INSTALL}, or use l1_solver='builtin'."
        )
    module = importlib.import_module("glmnet")
    distribution = importlib.metadata.distribution("glmstar")
    owned = {
        Path(str(distribution.locate_file(entry))).resolve()
        for entry in distribution.files or ()
        if str(entry).replace("\\", "/") == "glmnet/__init__.py"
    }
    if Path(str(module.__file__)).resolve() not in owned or not hasattr(module, "LogNet"):
        raise ImportError(
            f"l1_solver='glmstar' found a 'glmnet' module at {module.__file__} that is not "
            "glmstar's. Another package, such as python-glmnet, installs the same name. "
            f"Uninstall it and reinstall with {_INSTALL}."
        )
    paths = importlib.import_module("glmnet.paths.fastnet")
    return module.LogNet, module.MultiClassNet, paths.FastNetControl


def starts(problem: _Problem, penalties: FloatArray) -> list[tuple[FloatArray, FloatArray]]:
    """glmstar's solution at each penalty, in the built-in solver's parameterization.

    Parameters
    ----------
    problem : _Problem
        The standardized training problem.
    penalties : ndarray of float
        Positive penalties below the problem's null gradient, strictly decreasing.

    Returns
    -------
    list of tuple of (ndarray, ndarray)
        ``(intercept, coef)`` per penalty: ``(K,)`` and ``(p, K)``; for two classes the
        intercept is ``(0, b0)`` and the coefficients sit in column 1, as the built-in
        solver keeps them.

    Raises
    ------
    GlmstarFallback
        On glmstar's non-convergence warning with a shortened path, a fatal glmstar error
        code, a path shorter than the grid, non-finite output, or a class with no training
        row.
    ValueError
        When glmstar returns arrays of an unexpected shape.
    """
    if np.unique(problem.codes).size < problem.n_classes:
        # glmstar sizes its multinomial by the labels it sees, so a class with no training
        # row cannot be mapped back; the built-in solver keeps that class.
        raise GlmstarFallback("absent_class")
    log_net, multi_class_net, control_class = load()
    control = control_class(thresh=1e-18, maxit=10**8, fdev=0.0, devmax=1.0)
    p, n_classes = problem.p, problem.n_classes
    settings: dict[str, Any] = {
        "lambda_values": np.asarray(penalties, dtype=float),
        "standardize": False,
        "fit_intercept": True,
        "pmax": p + 1,
        "control": control,
    }
    estimator = (
        log_net(**settings) if problem.binary else multi_class_net(grouped=False, **settings)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            estimator.fit(np.asarray(problem.z), problem.codes, sample_weight=problem.w)
        except Exception as error:
            # A fatal code (a class share below glmnet's floor, memory) is logged, not raised,
            # and the extraction that follows can fail on the empty path.
            if _jerr(estimator) > 0:
                raise GlmstarFallback("fatal") from error
            if isinstance(error, (TypeError, AttributeError)):
                raise type(error)(
                    f"glmstar {importlib.metadata.version('glmstar')} failed inside its fit "
                    f"({error}). It may have changed its interface; reinstall with {_INSTALL}, "
                    "or use l1_solver='builtin'."
                ) from error
            raise
    if _jerr(estimator) > 0:
        raise GlmstarFallback("fatal")
    coefs = np.asarray(estimator.coefs_, dtype=float)
    intercepts = np.asarray(estimator.intercepts_, dtype=float)
    m = penalties.size
    if coefs.shape[0] < m:
        raise GlmstarFallback("nonconvergence" if caught else "short_path")
    expected = (m, p) if problem.binary else (m, p, n_classes)
    if coefs.shape != expected or intercepts.shape != expected[:1] + expected[2:]:
        raise ValueError(
            f"glmstar returned coefficients of shape {coefs.shape} and intercepts of shape "
            f"{intercepts.shape}; the adapter expects {expected}. glmstar "
            f"{importlib.metadata.version('glmstar')} may have changed its output layout."
        )
    if not (np.all(np.isfinite(coefs)) and np.all(np.isfinite(intercepts))):
        raise GlmstarFallback("nonfinite")
    out: list[tuple[FloatArray, FloatArray]] = []
    for k in range(m):
        coef = np.zeros((p, n_classes))
        if problem.binary:
            intercept = np.array([0.0, float(intercepts[k])])
            coef[:, 1] = coefs[k]
        else:
            intercept = intercepts[k].copy()
            coef[:, :] = coefs[k]
        out.append((intercept, coef))
    return out
