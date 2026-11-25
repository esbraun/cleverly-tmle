"""Bind each ``cleverly`` call a notebook makes to the current signature, without running it.

The TWINS notebook downloads its data, so the offline fast tier never executes it, and an API
rename that breaks one of its calls would go unseen until the next re-execution.  This module
parses the code cells, resolves each call whose target it can name statically, and binds the
call's arguments with :meth:`inspect.Signature.bind`.  A removed or renamed keyword, a missing
required argument, or a name the package no longer exports then fails in the fast tier.

What resolves: names imported from a ``cleverly`` module; attributes of those modules and
classes; and methods called on a name bound to a constructor call, or to a call whose return
annotation names a class.  Everything else (pandas, scikit-learn, values built in loops) is left
alone, because guessing its type would only add false failures.  A call that spreads ``*args``
or ``**kwargs`` cannot be bound statically and is skipped.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
from collections.abc import Iterable
from typing import Any

_UNRESOLVED = object()


def _import_names(tree: ast.AST, scope: dict[str, Any]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "cleverly" or alias.name.startswith("cleverly."):
                    module = importlib.import_module(alias.name)
                    if alias.asname:
                        scope[alias.asname] = module
                    else:
                        scope[alias.name.split(".")[0]] = importlib.import_module("cleverly")
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module != "cleverly" and not node.module.startswith("cleverly."):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                name = alias.asname or alias.name
                value = getattr(module, alias.name, _UNRESOLVED)
                if value is _UNRESOLVED:
                    value = importlib.import_module(f"{node.module}.{alias.name}")
                scope[name] = value


def _resolve(node: ast.expr, scope: dict[str, Any], types: dict[str, type]) -> Any:
    """The object ``node`` names, a ``("method", cls, name)`` triple, or :data:`_UNRESOLVED`."""
    if isinstance(node, ast.Name):
        return scope.get(node.id, _UNRESOLVED)
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in types:
            owner = types[node.value.id]
            if hasattr(owner, node.attr):
                return ("method", owner, node.attr)
            return ("missing", owner, node.attr)
        base = _resolve(node.value, scope, types)
        if base is _UNRESOLVED or isinstance(base, tuple):
            return _UNRESOLVED
        if inspect.ismodule(base) or inspect.isclass(base):
            return getattr(base, node.attr, ("missing", base, node.attr))
    return _UNRESOLVED


def _returned_class(target: Any) -> type | None:
    """The class a callable's return annotation names, if it names exactly one class."""
    if inspect.isclass(target):
        return target
    try:
        hints = typing.get_type_hints(target)
    except Exception:  # an unresolvable forward reference leaves the type unknown
        return None
    returned = hints.get("return")
    return returned if inspect.isclass(returned) else None


def _call_failure(call: ast.Call, target: Any) -> str | None:
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return None
    if isinstance(target, tuple) and target[0] == "missing":
        return f"{getattr(target[1], '__name__', target[1])} has no attribute {target[2]!r}"
    bound_self: tuple[Any, ...] = ()
    if isinstance(target, tuple):
        _, owner, name = target
        target = inspect.getattr_static(owner, name)
        if isinstance(target, property):
            return None
        if isinstance(target, staticmethod):
            target = target.__func__
        elif isinstance(target, classmethod):
            target = target.__func__
            bound_self = (owner,)
        else:
            bound_self = (None,)
    if not callable(target):
        return None
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    arguments = [None] * len(call.args)
    keywords = {keyword.arg: None for keyword in call.keywords if keyword.arg}
    try:
        signature.bind(*bound_self, *arguments, **keywords)
    except TypeError as error:
        return str(error)
    return None


def unbound_calls(cells: Iterable[tuple[str, str]]) -> list[str]:
    """Each ``cleverly`` call in ``cells`` that does not bind to the current signature.

    Parameters
    ----------
    cells : Iterable[tuple[str, str]]
        ``(cell id, source)`` pairs in notebook order.

    Returns
    -------
    list[str]
        ``"cell: line N: call: reason"`` for each call that fails to bind.
    """
    scope: dict[str, Any] = {}
    types: dict[str, type] = {}
    failures = []
    for cell_id, source in cells:
        tree = ast.parse(source)
        _import_names(tree, scope)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = _resolve(node.func, scope, types)
                if target is _UNRESOLVED:
                    continue
                reason = _call_failure(node, target)
                if reason:
                    failures.append(
                        f"{cell_id}: line {node.lineno}: {ast.unparse(node.func)}: {reason}"
                    )
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)
            ):
                target = _resolve(node.value.func, scope, types)
                if isinstance(target, tuple) and target[0] == "method":
                    target = getattr(target[1], target[2])
                returned = None if target is _UNRESOLVED else _returned_class(target)
                if returned is not None:
                    types[node.targets[0].id] = returned
                else:
                    types.pop(node.targets[0].id, None)
    return failures
