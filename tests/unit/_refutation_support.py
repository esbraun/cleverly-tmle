"""Shared stand-ins for the refutation unit tests that replace a real refit.

The bootstrap measurement-error and generated-outcome tests drive ``refute()`` through a
deterministic refit seam.  Each seam returns a :class:`RefitResult` holding one
:func:`stub_estimate`, and each test wraps its prepared rows in a :class:`StubResult`.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def stub_estimate(psi: float, standard_error: float) -> SimpleNamespace:
    """A stand-in for :class:`~cleverly.inference.ParameterEstimate`.

    It carries ``plugin_std_error`` beside ``std_error`` because the real class does: RM12
    split the two so a selector-path collaborative fit can refuse the inferential name and
    keep the diagnostic.  A stub with only ``std_error`` would pass here while the library
    reads the accessor it actually uses.
    """
    return SimpleNamespace(
        psi=psi,
        std_error=standard_error,
        plugin_std_error=standard_error,
        inference="influence_curve",
    )


class RefitResult:
    """What a seam's ``refit`` returns: the rows it fitted and its one ``ate`` estimate.

    ``estimate`` is public, so a test can wrap one refit's estimate in another result or
    move the estimate after the refit returns.
    """

    def __init__(self, data: Any, estimate: Any) -> None:
        self.data = data
        self.estimate = estimate

    def __getitem__(self, name: str) -> Any:
        assert name == "ate"
        return self.estimate


class StubResult(SimpleNamespace):
    """A fitted-result stand-in that answers ``result[name]`` from ``estimates``."""

    def __getitem__(self, name: str) -> Any:
        return self.estimates[name]
