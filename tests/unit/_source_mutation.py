"""Apply a deliberate source mutation to one function, through ``monkeypatch`` only.

A mutation control asks a check to fail against a broken implementation.  The broken
implementation here is the shipped function's own source with one anchored edit, compiled in
the namespace of the module that owns it, so the control breaks exactly one line and touches no
file.  ``monkeypatch`` restores the original at the end of the test.
"""

from __future__ import annotations

import inspect
import textwrap
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest


def mutate(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    owner: Any,
    name: str,
    edits: Sequence[tuple[str, str]],
) -> None:
    """Replace ``owner.name`` with its own source after ``edits``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The fixture that restores the original.
    module : ModuleType
        The module whose namespace the function reads its globals from.
    owner : object
        The class or module that holds the function.
    name : str
        The function's name on ``owner``.
    edits : sequence of tuple of str
        Each ``(old, new)`` pair.  ``old`` must occur exactly once, so an anchor that drifts
        with a code change fails loudly instead of mutating nothing.
    """
    source = inspect.getsource(getattr(owner, name))
    for old, new in edits:
        count = source.count(old)
        assert count == 1, f"the mutation anchor occurs {count} times in {name}: {old!r}"
        source = source.replace(old, new)
    namespace: dict[str, Any] = {}
    code = compile(
        "from __future__ import annotations\n" + textwrap.dedent(source),
        f"<mutated {name}>",
        "exec",
    )
    exec(code, vars(module), namespace)
    monkeypatch.setattr(owner, name, namespace[name])
