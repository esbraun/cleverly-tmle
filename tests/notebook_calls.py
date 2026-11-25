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
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import sys
import types
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


#: The classes a value may be: one class, or each member of a union annotation.
Classes = tuple[type, ...]


@functools.cache
def _deferred_imports(module: types.ModuleType) -> dict[str, str]:
    """The names a module imports only for type checking, mapped to their absolute module."""
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError):
        return {}
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module is not None:
                    package = module.__package__ or ""
                    absolute = (
                        importlib.util.resolve_name("." * child.level + child.module, package)
                        if child.level
                        else child.module
                    )
                    names.update({alias.asname or alias.name: absolute for alias in child.names})
    return names


def _class_named(name: str, home: Any) -> type | None:
    """The class an annotation names: in ``home``'s module, else the one ``cleverly`` class."""
    name = name.strip().strip("'\"")
    if not name.isidentifier():
        return None
    module = sys.modules.get(getattr(home, "__module__", ""), None)
    found = getattr(module, name, None)
    if inspect.isclass(found):
        return found
    if module is not None and name in _deferred_imports(module):
        found = getattr(importlib.import_module(_deferred_imports(module)[name]), name, None)
        if inspect.isclass(found):
            return found
    matches = {
        id(value): value
        for key, loaded in list(sys.modules.items())
        if key == "cleverly" or key.startswith("cleverly.")
        for value in [getattr(loaded, name, None)]
        if inspect.isclass(value)
    }
    return next(iter(matches.values())) if len(matches) == 1 else None


def _annotated_classes(annotation: Any, home: Any) -> Classes | None:
    """The classes an annotation names, or None unless every member names a class."""
    if annotation in (inspect.Signature.empty, None, typing.Any, "Any"):
        return None
    if inspect.isclass(annotation):
        return (annotation,)
    if isinstance(annotation, str):
        members = [_class_named(part, home) for part in annotation.split("|")]
    elif typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = [arg if inspect.isclass(arg) else None for arg in typing.get_args(annotation)]
    else:
        return None
    if any(member is None or member is type(None) for member in members):
        return None
    return tuple(member for member in members if member is not None)


def _returned_classes(target: Any) -> Classes | None:
    """The classes a callable's return annotation names.

    A forward reference that ``typing.get_type_hints`` cannot evaluate, such as a name imported
    only for type checking, is looked up by name among the loaded ``cleverly`` modules.
    """
    if inspect.isclass(target):
        return (target,)
    if isinstance(target, property):
        target = target.fget
    elif isinstance(target, functools.cached_property):
        target = target.func
    try:
        returned = typing.get_type_hints(target).get("return")
    except Exception:
        try:
            returned = inspect.signature(target).return_annotation
        except (TypeError, ValueError):
            return None
    return _annotated_classes(returned, target)


def _field_classes(owner: type, attribute: str) -> Classes | None:
    """The classes of an instance attribute: a property's return, or a field's annotation."""
    static = inspect.getattr_static(owner, attribute, None)
    if isinstance(static, (property, functools.cached_property)):
        return _returned_classes(static)
    for klass in owner.__mro__:
        annotation = inspect.get_annotations(klass).get(attribute)
        if annotation is not None:
            return _annotated_classes(annotation, klass)
    return None


def _has_attribute(owner: type, attribute: str) -> bool | None:
    """Whether instances of ``owner`` have ``attribute``; None when that cannot be known."""
    if hasattr(owner, attribute) or any(
        attribute in inspect.get_annotations(klass) for klass in owner.__mro__
    ):
        return True
    if any("__getattr__" in klass.__dict__ for klass in owner.__mro__[:-1]):
        return None
    if not dataclasses.is_dataclass(owner) and not hasattr(owner, "__slots__"):
        return None  # a plain class may set the attribute in __init__
    return False


def _instance_types(node: ast.expr, bound: dict[str, Classes]) -> Classes | None:
    """The classes of the value ``node`` names: a typed name, or a typed attribute chain."""
    if isinstance(node, ast.Name):
        return bound.get(node.id)
    if isinstance(node, ast.Attribute):
        owners = _instance_types(node.value, bound)
        if owners is None:
            return None
        # A union member whose attribute has no class annotation (``Any``) is left out, so the
        # members that do declare one still check the calls made on the attribute.
        found: list[type] = []
        for owner in owners:
            if not _has_attribute(owner, node.attr):
                continue
            classes = _field_classes(owner, node.attr) or ()
            found.extend(klass for klass in classes if klass not in found)
        return tuple(found) or None
    return None


def _resolve(node: ast.expr, scope: dict[str, Any], bound: dict[str, Classes]) -> Any:
    """The object ``node`` names, a ``("method", classes, name)`` triple, or :data:`_UNRESOLVED`.

    On a value whose type is a union, the method resolves on the members that define it, and
    ``("missing", classes, name)`` means no member does.
    """
    if isinstance(node, ast.Name):
        return scope.get(node.id, _UNRESOLVED)
    if isinstance(node, ast.Attribute):
        owners = _instance_types(node.value, bound)
        if owners is not None:
            known = {owner: _has_attribute(owner, node.attr) for owner in owners}
            having = tuple(owner for owner, has in known.items() if has)
            if having:
                return ("method", having, node.attr)
            if all(has is False for has in known.values()):
                return ("missing", owners, node.attr)
            return _UNRESOLVED
        base = _resolve(node.value, scope, bound)
        if base is _UNRESOLVED or isinstance(base, tuple):
            return _UNRESOLVED
        if inspect.ismodule(base) or inspect.isclass(base):
            if hasattr(base, node.attr):
                return getattr(base, node.attr)
            return ("missing", (base,), node.attr)
    return _UNRESOLVED


def _bind_failure(call: ast.Call, target: Any, bound_self: tuple[Any, ...]) -> str | None:
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


def _method_failure(call: ast.Call, owner: type, name: str) -> str | None:
    target = inspect.getattr_static(owner, name, None)
    if target is None or isinstance(target, (property, functools.cached_property)):
        return None  # an instance field, or a property that returns a callable
    if isinstance(target, staticmethod):
        return _bind_failure(call, target.__func__, ())
    if isinstance(target, classmethod):
        return _bind_failure(call, target.__func__, (owner,))
    return _bind_failure(call, target, (None,))


def _call_failure(call: ast.Call, target: Any) -> str | None:
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return None
    if isinstance(target, tuple):
        kind, owners, name = target
        names = " | ".join(getattr(owner, "__name__", str(owner)) for owner in owners)
        if kind == "missing":
            return f"{names} has no attribute {name!r}"
        # A union member that accepts the call is enough: the notebook may hold any of them.
        reasons = [_method_failure(call, owner, name) for owner in owners]
        return None if any(reason is None for reason in reasons) else reasons[0]
    return _bind_failure(call, target, ())


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
    bound: dict[str, Classes] = {}
    failures = []
    for cell_id, source in cells:
        tree = ast.parse(source)
        _import_names(tree, scope)
        # Statements in source order, so a call sees the types its earlier lines bound.
        for statement in tree.body:
            for node in ast.walk(statement):
                if not isinstance(node, ast.Call):
                    continue
                target = _resolve(node.func, scope, bound)
                if target is _UNRESOLVED:
                    continue
                reason = _call_failure(node, target)
                if reason:
                    failures.append(
                        f"{cell_id}: line {node.lineno}: {ast.unparse(node.func)}: {reason}"
                    )
            if isinstance(statement, ast.Assign):
                _bind_types(statement, scope, bound)
    return failures


def _bind_types(statement: ast.Assign, scope: dict[str, Any], bound: dict[str, Classes]) -> None:
    """Record the classes an assignment binds to a name, or forget the name's old classes."""
    names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
    returned: Classes | None = None
    if isinstance(statement.value, ast.Call):
        target = _resolve(statement.value.func, scope, bound)
        if isinstance(target, tuple) and target[0] == "method":
            found: list[type] = []
            for owner in target[1]:
                method = inspect.getattr_static(owner, target[2])
                if isinstance(method, (staticmethod, classmethod)):
                    method = method.__func__
                classes = _returned_classes(method)
                if classes is None:
                    found = []
                    break
                found.extend(klass for klass in classes if klass not in found)
            returned = tuple(found) or None
        elif target is not _UNRESOLVED and not isinstance(target, tuple):
            returned = _returned_classes(target)
    elif isinstance(statement.value, ast.Attribute):
        returned = _instance_types(statement.value, bound)
    for name in names:
        if returned is not None:
            bound[name] = returned
        else:
            bound.pop(name, None)
