"""Every refusal names its reason code, and the code survives every route a refusal takes.

The checks read the source with the AST rather than importing it, because ``import cleverly``
does not import every module that defines an error class (``cleverly.sensitivity.evalue``
defines ``_EValueRefusal`` and is imported lazily). A class that derives from an error class
is found by closing over the ``ClassDef`` bases of the whole package.
"""

from __future__ import annotations

import ast
import pickle
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from cleverly._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from cleverly.exceptions import (
    CapabilityError,
    CleverlyError,
    DataError,
    LongitudinalError,
    MethodConfigurationError,
    NotFittedError,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "cleverly"
ROADMAP = ROOT / "docs" / "roadmap.md"

#: The error classes whose raises must name a code: they declare no default.
CODED_ROOTS = frozenset({"CapabilityError"})
#: Every error class of the package.
ERROR_ROOTS = frozenset({"CleverlyError"})


@dataclass(frozen=True)
class Finding:
    """One construction that breaks a rule, with where it is."""

    path: str
    line: int
    rule: str


def _modules(root: Path = SOURCE) -> Iterator[tuple[str, ast.Module]]:
    for path in sorted(root.rglob("*.py")):
        yield str(path.relative_to(ROOT)), ast.parse(path.read_text(encoding="utf-8"))


def _base_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def subclasses(modules: list[tuple[str, ast.Module]], roots: frozenset[str]) -> frozenset[str]:
    """The names of every class that derives from one of ``roots``, the roots included.

    Parameters
    ----------
    modules : list of tuple of str and ast.Module
        The parsed package.
    roots : frozenset of str
        The class names to close over.

    Returns
    -------
    frozenset of str
        ``roots`` and the name of every class whose bases reach one of them.
    """
    bases: dict[str, set[str]] = {}
    for _, module in modules:
        for node in ast.walk(module):
            if isinstance(node, ast.ClassDef):
                names = {name for base in node.bases if (name := _base_name(base))}
                bases.setdefault(node.name, set()).update(names)
    found = set(roots)
    changed = True
    while changed:
        changed = False
        for name, parents in bases.items():
            if name not in found and parents & found:
                found.add(name)
                changed = True
    return frozenset(found)


def _keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((keyword for keyword in call.keywords if keyword.arg == name), None)


def _keyword_only_without_default(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    arguments = function.args
    return {
        argument.arg
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True)
        if default is None
    }


def construction_findings(modules: list[tuple[str, ast.Module]]) -> list[Finding]:
    """Every construction of a coded error class that names no code.

    A coded class (one with no default code) is built with ``reason=`` or ``refusal=``, or
    through ``from_refusal``. When ``reason=`` passes a parameter of the enclosing function,
    that parameter is keyword-only with no default, so every caller of the helper names a code.

    Parameters
    ----------
    modules : list of tuple of str and ast.Module
        The parsed package.

    Returns
    -------
    list of Finding
        One entry per construction that breaks the rule.
    """
    coded = subclasses(modules, CODED_ROOTS)
    findings: list[Finding] = []
    for path, module in modules:
        for function in [module, *ast.walk(module)]:
            if not isinstance(function, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            body = function.body if isinstance(function, ast.Module) else [function]
            required = (
                set()
                if isinstance(function, ast.Module)
                else _keyword_only_without_default(function)
            )
            parameters = (
                set()
                if isinstance(function, ast.Module)
                else {
                    argument.arg
                    for argument in ast.walk(function.args)
                    if isinstance(argument, ast.arg)
                }
            )
            for statement in body:
                for node in ast.walk(statement):
                    if not isinstance(node, ast.Call) or _base_name(node.func) not in coded:
                        continue
                    if isinstance(node.func, ast.Attribute) and node.func.attr not in coded:
                        continue
                    if _keyword(node, "refusal") is not None:
                        continue
                    reason = _keyword(node, "reason")
                    if reason is None:
                        findings.append(Finding(path, node.lineno, "names no reason="))
                    elif (
                        isinstance(reason.value, ast.Name)
                        and reason.value.id in parameters
                        and reason.value.id not in required
                    ):
                        findings.append(
                            Finding(path, node.lineno, f"{reason.value.id} has a default")
                        )
    return sorted(set(findings), key=lambda finding: (finding.path, finding.line))


def roadmap_ids(modules: list[tuple[str, ast.Module]]) -> list[tuple[str, int, ast.expr]]:
    """Every ``roadmap=`` value written in the package, with where it is.

    Parameters
    ----------
    modules : list of tuple of str and ast.Module
        The parsed package.

    Returns
    -------
    list of tuple of str, int and ast.expr
        The path, the line, and the value of each ``roadmap=`` keyword.
    """
    values = []
    for path, module in modules:
        for node in ast.walk(module):
            if isinstance(node, ast.Call):
                keyword = _keyword(node, "roadmap")
                if keyword is not None:
                    values.append((path, node.lineno, keyword.value))
    return values


#: The built-in classes a malformed argument raises. A capability refusal raises
#: ``CapabilityError`` with a code instead.
BUILTIN_REFUSALS = frozenset({"ValueError", "TypeError", "NotImplementedError"})

#: The wording of a refusal. A built-in raise whose text matches it either refuses a
#: capability, which needs a code, or names a malformed argument in refusal wording, which
#: the allow list below records with its reason.
REFUSAL_VOCABULARY = re.compile(
    r"is not supported|are not supported|is refused|refused|not implemented"
    r"|cannot be estimated|no derivation|use .* instead",
    re.IGNORECASE | re.DOTALL,
)

#: Every built-in raise that uses refusal wording, keyed by the file and a fragment of its
#: text. The value says why the raise reports a malformed argument and refuses no capability.
BUILTIN_ALLOWED: dict[tuple[str, str], str] = {
    (
        "src/cleverly/datasets/synthetic.py",
        "This is refused rather than clipped",
    ): "make_* generator argument: an outcome_mean outside the probability range is a "
    "malformed argument of a data generator, and no estimator capability is involved",
    **{
        (path, "_SECOND_MOMENT_CLUSTER_REFUSAL"): "low-level guard on a covariance_rule and "
        "cluster pair: every fit that declares the second-moment rule (the stacked "
        "natural-course mean) refuses id= upstream with a coded CapabilityError, so only a "
        "direct call with a malformed pair reaches it"
        for path in ("src/cleverly/inference/influence.py", "src/cleverly/inference/results.py")
    },
}


@dataclass(frozen=True)
class BuiltinHit:
    """One built-in raise whose text reads as a refusal, with where it is."""

    path: str
    line: int
    kind: str
    text: str


def builtin_refusal_hits(modules: list[tuple[str, ast.Module]]) -> list[BuiltinHit]:
    """Every built-in raise whose literal text matches the refusal vocabulary.

    The text is every string constant in the call, f-string parts included. An argument
    that is a bare name ending in ``REFUSAL`` is a hit too, because the text of a refusal
    constant is a refusal whatever it says.

    Parameters
    ----------
    modules : list of tuple of str and ast.Module
        The parsed package.

    Returns
    -------
    list of BuiltinHit
        One entry per matching raise.
    """
    hits = []
    for path, module in modules:
        for node in ast.walk(module):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            call = node.exc
            if not (isinstance(call.func, ast.Name) and call.func.id in BUILTIN_REFUSALS):
                continue
            text = "".join(
                part.value
                for argument in call.args
                for part in ast.walk(argument)
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            names = [
                argument.id
                for argument in call.args
                if isinstance(argument, ast.Name) and argument.id.endswith("REFUSAL")
            ]
            if REFUSAL_VOCABULARY.search(text) or names:
                hits.append(BuiltinHit(path, node.lineno, call.func.id, text or names[0]))
    return hits


def unreviewed_builtin_hits(
    hits: list[BuiltinHit], allowed: dict[tuple[str, str], str]
) -> list[BuiltinHit]:
    """The hits that no allow-list entry covers.

    Parameters
    ----------
    hits : list of BuiltinHit
        The built-in raises that read as a refusal.
    allowed : dict
        The allow list, keyed by file and text fragment.

    Returns
    -------
    list of BuiltinHit
        The hits whose file and text match no key.
    """
    return [
        hit
        for hit in hits
        if not any(hit.path == path and fragment in hit.text for path, fragment in allowed)
    ]


@pytest.fixture(scope="module")
def modules() -> list[tuple[str, ast.Module]]:
    return list(_modules())


class TestTheConstructionRule:
    def test_every_coded_error_names_its_reason(
        self, modules: list[tuple[str, ast.Module]]
    ) -> None:
        findings = construction_findings(modules)
        assert not findings, "\n".join(f"{f.path}:{f.line}: {f.rule}" for f in findings)

    def test_the_rule_sees_a_lazily_imported_subclass(
        self, modules: list[tuple[str, ast.Module]]
    ) -> None:
        assert "_EValueRefusal" in subclasses(modules, CODED_ROOTS)

    def test_the_rule_fails_a_bare_raise(self) -> None:
        source = ast.parse(
            "class CapabilityError(Exception): ...\n"
            "class Sub(CapabilityError): ...\n"
            "def f():\n    raise Sub('x')\n"
        )
        assert construction_findings([("synthetic.py", source)]) == [
            Finding("synthetic.py", 4, "names no reason=")
        ]

    def test_the_rule_fails_a_helper_whose_code_has_a_default(self) -> None:
        source = ast.parse(
            "class CapabilityError(Exception): ...\n"
            "def refuse(*, reason=None):\n    raise CapabilityError('x', reason=reason)\n"
        )
        assert construction_findings([("synthetic.py", source)]) == [
            Finding("synthetic.py", 3, "reason has a default")
        ]

    def test_every_roadmap_id_is_a_literal_with_a_heading(
        self, modules: list[tuple[str, ast.Module]]
    ) -> None:
        roadmap = ROADMAP.read_text(encoding="utf-8")
        problems = []
        for path, line, value in roadmap_ids(modules):
            if isinstance(value, ast.Constant) and value.value is None:
                continue
            if isinstance(value, ast.Name | ast.Attribute):
                # A helper forwards its caller's id, and a status record holds its own.
                continue
            if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
                problems.append(f"{path}:{line}: roadmap= is neither a string literal nor a name")
                continue
            heading = re.compile(rf"^#{{2,4}} {re.escape(value.value)}\. ", re.MULTILINE)
            if not heading.search(roadmap):
                problems.append(f"{path}:{line}: {value.value} has no heading in docs/roadmap.md")
        assert not problems, "\n".join(problems)


class TestTheRecord:
    @pytest.mark.parametrize(
        "reason",
        [reason for reason in RefusalReason if reason is not RefusalReason.INVALID_DECLARATION],
    )
    def test_a_code_other_than_invalid_declaration_needs_an_alternative(
        self, reason: RefusalReason
    ) -> None:
        with pytest.raises(TypeError, match="alternative"):
            Refusal(reason, "The condition failed.")

    def test_an_invalid_declaration_needs_no_alternative(self) -> None:
        refusal = Refusal(RefusalReason.INVALID_DECLARATION, "The column is missing.")
        assert refusal.message() == "The column is missing."

    def test_the_message_names_the_roadmap_item_once(self) -> None:
        named = Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "F21 in docs/roadmap.md reopens this.",
            alternative="Fit it in sample.",
            roadmap="F21",
        )
        assert named.message().count("F21") == 1
        unnamed = Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "The fold-specific update is not written.",
            alternative="Fit it in sample.",
            roadmap="F21",
        )
        assert unnamed.message() == (
            "The fold-specific update is not written. F21 in docs/roadmap.md tracks this. "
            "Fit it in sample."
        )

    def test_no_alternative_is_a_sentence(self) -> None:
        assert NO_ALTERNATIVE.endswith(".")


class TestTheErrors:
    @pytest.mark.parametrize(
        "cls",
        [DataError, MethodConfigurationError, LongitudinalError, NotFittedError, CleverlyError],
    )
    def test_a_class_with_a_default_takes_invalid_declaration(
        self, cls: type[CleverlyError]
    ) -> None:
        assert cls("The column is missing.").refusal.reason is RefusalReason.INVALID_DECLARATION

    def test_a_capability_error_needs_a_code(self) -> None:
        with pytest.raises(TypeError, match="reason="):
            CapabilityError("The composition is not written.")

    def test_the_text_is_the_record_s_message(self) -> None:
        error = CapabilityError(
            "The fold update is not written.",
            reason=RefusalReason.API_REDIRECTION,
            alternative="Use targeting_scheme='pooled'.",
        )
        assert str(error) == error.refusal.message()

    def test_a_record_and_its_parts_do_not_mix(self) -> None:
        refusal = Refusal(RefusalReason.INVALID_DECLARATION, "x.")
        with pytest.raises(TypeError, match="not both"):
            CapabilityError("y.", refusal=refusal)

    @pytest.mark.parametrize(
        "error",
        [
            CapabilityError(
                "The fold update is not written.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Fit it in sample.",
                roadmap="F21",
            ),
            DataError("The column is missing."),
            NotFittedError("Call fit first."),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_the_record_survives_pickle(self, error: CleverlyError) -> None:
        loaded = pickle.loads(pickle.dumps(error))
        assert type(loaded) is type(error)
        assert loaded.refusal == error.refusal
        assert str(loaded) == str(error)

    def test_a_subclass_with_its_own_constructor_survives_pickle(self) -> None:
        from cleverly.assessment import AssessmentStatus
        from cleverly.sensitivity.evalue import _EValueRefusal

        error = _EValueRefusal(
            AssessmentStatus.UNAVAILABLE,
            "The E-value needs a ratio.",
            reason=RefusalReason.INFERENCE_GAP,
            alternative="Request rr.",
        )
        loaded = pickle.loads(pickle.dumps(error))
        assert loaded.status == error.status
        assert loaded.refusal == error.refusal

    def test_the_record_survives_a_worker_re_raise(self) -> None:
        from joblib import Parallel, delayed

        def refuse() -> None:
            raise CapabilityError(
                "The fold update is not written.",
                reason=RefusalReason.IMPLEMENTATION_GAP,
                alternative="Fit it in sample.",
            )

        with pytest.raises(CapabilityError) as raised:
            Parallel(n_jobs=2, backend="loky")(delayed(refuse)() for _ in range(2))
        assert raised.value.refusal.reason is RefusalReason.IMPLEMENTATION_GAP


class TestTheBuiltinRefusalScan:
    def test_every_builtin_that_reads_as_a_refusal_is_a_reviewed_malformed_argument(
        self, modules: list[tuple[str, ast.Module]]
    ) -> None:
        unreviewed = unreviewed_builtin_hits(builtin_refusal_hits(modules), BUILTIN_ALLOWED)
        assert not unreviewed, "\n".join(
            f"{hit.path}:{hit.line}: {hit.kind}({hit.text[:80]!r}) reads as a capability "
            "refusal. Raise CapabilityError with a reason code, or record why it is a "
            "malformed argument in BUILTIN_ALLOWED"
            for hit in unreviewed
        )

    def test_every_allow_list_entry_matches_a_hit_and_states_a_reason(
        self, modules: list[tuple[str, ast.Module]]
    ) -> None:
        hits = builtin_refusal_hits(modules)
        stale = [
            key
            for key in BUILTIN_ALLOWED
            if not any(hit.path == key[0] and key[1] in hit.text for hit in hits)
        ]
        assert not stale, f"allow-list entries that match no raise: {stale}"
        assert all(reason.strip() for reason in BUILTIN_ALLOWED.values())

    def test_the_scan_sees_a_known_hit(self, modules: list[tuple[str, ast.Module]]) -> None:
        paths = {hit.path for hit in builtin_refusal_hits(modules)}
        assert "src/cleverly/datasets/synthetic.py" in paths

    @pytest.mark.parametrize(
        "raised",
        [
            'raise ValueError("x is not supported")',
            'raise TypeError("the option is refused")',
            'raise NotImplementedError("not implemented here")',
            'raise ValueError(f"{name} cannot be estimated for {kind}")',
            'raise ValueError("a" "b no derivation")',
            'raise ValueError("this fails; use y instead")',
            "raise ValueError(SOME_REFUSAL)",
        ],
    )
    def test_the_scan_fails_a_bare_builtin_refusal(self, raised: str) -> None:
        source = ast.parse(f"def f(name, kind):\n    {raised}\n")
        hits = builtin_refusal_hits([("synthetic.py", source)])
        assert len(hits) == 1
        assert unreviewed_builtin_hits(hits, BUILTIN_ALLOWED) == hits

    def test_the_scan_passes_a_reviewed_entry_and_a_plain_argument_error(self) -> None:
        source = ast.parse(
            "def f():\n"
            "    raise ValueError('x is not supported')\n"
            "def g():\n"
            "    raise ValueError('alpha must lie in (0, 1)')\n"
            "def h():\n"
            "    raise CapabilityError('x is not supported', reason=None)\n"
        )
        hits = builtin_refusal_hits([("synthetic.py", source)])
        assert [hit.line for hit in hits] == [2]
        allowed = {("synthetic.py", "x is not supported"): "a malformed argument"}
        assert unreviewed_builtin_hits(hits, allowed) == []
        assert unreviewed_builtin_hits(hits, {("other.py", "x is not supported"): "r"}) == hits
