"""Which exceptions a repeated-sampling study records as a failed replicate.

A study that refits an estimator thousands of times must not lose its run to one draw on which
a fit breaks down numerically.  It must not hide a bug either: a programming error that fails
every replicate, recorded as "failures", would publish a study that measured nothing.  So the
rule records an explicit list of numerical and fitting failures and lets everything else stop
the study.

The rule reads the exception itself, never its ``__cause__`` or ``__context__``: an explicit
``raise X from e`` is a deliberate translation into ``X``.  It runs where the fit ran, because
a process pool re-raises an exception in the parent with its own frames innermost.

One limit is known.  A library check that detects an upstream ``cleverly`` defect, such as a
scikit-learn ``check_array`` on a NaN that ``cleverly`` produced, is judged by the module that
detected it.  Such a defect fails every replicate it touches, so a study's systematic stop and
its declared failure bound catch a constant cause.
"""

from __future__ import annotations

from types import TracebackType

import numpy as np

from ..exceptions import CapabilityError, DataError, LongitudinalError, MethodConfigurationError

__all__ = ["FATAL", "LIBRARY_MODULES", "RECORDED", "failure_call", "recordable"]

#: Exceptions that always stop the study.  Checked first, because ``CapabilityError`` and
#: ``MethodConfigurationError`` subclass ``ValueError``.
FATAL: tuple[type[BaseException], ...] = (
    KeyboardInterrupt,
    SystemExit,
    MemoryError,
    CapabilityError,
    MethodConfigurationError,
    TypeError,
    AttributeError,
    NameError,
    ImportError,
    AssertionError,
)

#: Exceptions recorded wherever they were raised.
RECORDED: tuple[type[BaseException], ...] = (
    DataError,
    LongitudinalError,
    np.linalg.LinAlgError,
    FloatingPointError,
    ZeroDivisionError,
    OverflowError,
)

#: Packages whose own plain ``ValueError`` or ``RuntimeError`` is a fitting failure.
LIBRARY_MODULES: tuple[str, ...] = ("numpy", "scipy", "sklearn", "statsmodels")


def _innermost(traceback: TracebackType | None) -> TracebackType | None:
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback


def _module(traceback: TracebackType | None) -> str:
    innermost = _innermost(traceback)
    if innermost is None:
        return ""
    return str(innermost.tb_frame.f_globals.get("__name__", ""))


def recordable(error: BaseException) -> bool:
    """Whether a study records ``error`` as a failed replicate rather than stopping.

    Parameters
    ----------
    error : BaseException
        The exception a fit raised, with its traceback.

    Returns
    -------
    bool
        True for a listed numerical or fitting failure, and for a plain ``ValueError`` or
        ``RuntimeError`` whose innermost frame is in a listed library.  False otherwise.
    """
    if isinstance(error, FATAL):
        return False
    if isinstance(error, RECORDED):
        return True
    if type(error) in (ValueError, RuntimeError):
        module = _module(error.__traceback__)
        return any(module == name or module.startswith(f"{name}.") for name in LIBRARY_MODULES)
    return False


def failure_call(error: BaseException) -> str:
    """``module:function`` of the frame that raised ``error``, or an empty string.

    Parameters
    ----------
    error : BaseException
        The exception a fit raised, with its traceback.

    Returns
    -------
    str
        The innermost frame's module and function name.
    """
    innermost = _innermost(error.__traceback__)
    if innermost is None:
        return ""
    frame = innermost.tb_frame
    return f"{frame.f_globals.get('__name__', '')}:{frame.f_code.co_name}"
