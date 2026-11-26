r"""Which documentation pages a change affects, and whether each was updated or waived.

Run it before you open a pull request; CI runs it on every pull request::

    git fetch origin main
    python -m tests.docs_impact --worktree --body-file body.md

``body.md`` is your draft pull request body, which carries the waivers. Without it the run lists
what to review and fails on every page you have not changed.

A page is **affected** for one of three reasons, and two more findings stand on their own:

1. **Routed.** The change touches a file that an area of :data:`tests.docs_map.AREAS` lists,
   and the page is one of that area's ``pages``. The pages under ``see_also`` are listed for the
   reviewer and never fail. The map is the union of the base branch's map and this change's, so
   a change cannot drop a page from the map that judges it.
2. **Quotes a removed name.** The change removes a public name from ``src/cleverly`` (an
   exported name, a public function or class, a public method, property or field, a public
   parameter, or a string option of a ``Literal`` annotation) that no module of the package
   still offers, and the page still quotes it inside backticks or as ``cleverly.<name>``.
3. **Quotes a changed default.** A public parameter or field changed its default, and the page
   quotes ``name=<old default>`` inside backticks.
4. **An undocumented addition.** The change adds a public name that no module offered before
   and no documentation source mentions.
5. **A problem with the waivers**: malformed, too short, naming an unknown page, or naming a
   page or name the change does not affect.

A routed page is satisfied when the change edits it beyond whitespace. A page that quotes a
removed name or an old default is satisfied when it no longer does. An addition is satisfied
when some page mentions it. Anything else needs a waiver line in the pull request body::

    Docs-waiver: docs/user-guide/settings.md: the new field is private and unreachable from the API
    Docs-waiver: cluster_floor: internal knob, set only by the study harness and never documented
    Docs-waiver: all: pure refactor, every public output and message is byte-identical

A reason needs at least four words. ``all`` covers routed pages only, never a quoted removed
name, an old default or an addition. Waivers inside a fenced code block or an HTML comment do
not count, so the pull request template can show the form.

**This reads paths, names and stated reasons, never prose.** Touching a page satisfies the
routing; it does not show that the page is now right. The pre-PR docs review in
``docs/development/pull-requests.md`` and the implementation review do that, and
``docs/architecture-invariants.md`` says why this is not a prose gate.

Standard library only, so the ``docs-impact`` CI job needs no install.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import fnmatch
import json
import os
import re
import secrets
import subprocess
import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tests.docs_map import AREAS, PUBLISHED, Area

__all__ = [
    "Finding",
    "Report",
    "Surface",
    "doc_pages",
    "evaluate",
    "main",
    "parse_waivers",
    "route",
    "surface",
]

ROOT = Path(__file__).resolve().parents[1]

WAIVER_KEY = re.compile(r"^[\s>*+\-\d.\[\]x]*docs-waiver\b", re.IGNORECASE)
WAIVER = re.compile(
    r"^\s*(?:[-*+]\s+)?docs-waiver:\s*`?(?P<target>[^\s:`]+)`?\s*:\s*(?P<reason>.*?)\s*$",
    re.IGNORECASE,
)
MIN_REASON_WORDS = 4
FENCE = re.compile(r"^\s*(```|~~~)")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

#: Pages that record history: a removed name is expected to appear in them.
HISTORY_PAGES = re.compile(r"^docs/(roadmap-delivered\.md|development/release-notes-.*\.md)$")
#: Study pages generated from committed results: they cannot be fixed by hand, and a study
#: regeneration rewrites them (``tests/unit/test_method_evidence.py`` gates their numbers).
GENERATED_PAGES = re.compile(r"^docs/technical-reference/method-evidence/(?!how-to-read)")
#: Pages that describe planned work: a mention there does not document a shipped name.
PLANNED_PAGES = frozenset({"docs/roadmap.md"})
EXCLUDED_PARTS = frozenset({"_build", "generated", ".ipynb_checkpoints"})
#: A shorter name matches too much prose to be worth reporting.
MIN_NAME_LENGTH = 3


def git(*args: str, root: Path = ROOT) -> str:
    """Run ``git`` in ``root`` and return its standard output; raise when it fails."""
    done = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def show(revision: str, path: str, root: Path = ROOT) -> str | None:
    """The text of ``path`` at ``revision``, or ``None`` when it did not exist there."""
    done = subprocess.run(
        ["git", "-C", str(root), "show", f"{revision}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout if done.returncode == 0 else None


# ---- the change -----------------------------------------------------------------------------


@dataclass
class Change:
    """What a change touched.

    Parameters
    ----------
    merge_base : str
        The commit the change branched from.
    paths : set of str
        Every path added, edited or removed; a rename gives both paths.
    edited : set of str
        The paths whose content changed beyond whitespace.
    worktree : bool
        Whether the head side is the working tree rather than ``head``.
    head : str, default "HEAD"
        The head revision when the head side is not the working tree.
    """

    merge_base: str
    paths: set[str]
    edited: set[str]
    worktree: bool
    head: str = "HEAD"

    def head_text(self, path: str, root: Path = ROOT) -> str | None:
        """The head version of ``path``, or ``None`` when the change deletes it."""
        if self.worktree:
            file = root / path
            return file.read_text(encoding="utf-8", errors="replace") if file.exists() else None
        return show(self.head, path, root)


def changed(base: str, *, worktree: bool, head: str = "HEAD", root: Path = ROOT) -> Change:
    """Collect the change from the merge base of ``base`` and ``HEAD`` to the head side.

    Rename detection is off, so a moved file reports its old path as well as its new one.
    """
    merge_base = git("merge-base", base, head, root=root).strip()
    target = [] if worktree else [head]
    names = git("diff", "--no-renames", "--name-only", merge_base, *target, root=root)
    stat = git(
        "diff",
        "--no-renames",
        "-w",
        "--ignore-blank-lines",
        "--numstat",
        merge_base,
        *target,
        root=root,
    )
    paths = {line for line in names.splitlines() if line}
    edited = set()
    for line in stat.splitlines():
        added, removed, path = line.split("\t", 2)
        if (added, removed) != ("0", "0"):
            edited.add(path)
    if worktree:
        untracked = git("ls-files", "--others", "--exclude-standard", root=root).splitlines()
        paths.update(untracked)
        edited.update(untracked)
    return Change(merge_base, paths, edited, worktree, head)


def parse_waivers(body: str) -> tuple[dict[str, str], list[str]]:
    """The waivers in a pull request body, and the problems with malformed waiver lines.

    Lines inside fenced code blocks and HTML comments are skipped.
    """
    waivers: dict[str, str] = {}
    problems: list[str] = []
    in_fence = False
    for line in COMMENT.sub("", body.replace("\r", "")).splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not WAIVER_KEY.match(line):
            continue
        match = WAIVER.match(line)
        if match is None:
            problems.append("a Docs-waiver line does not read 'Docs-waiver: <target>: <reason>'")
            continue
        target = match["target"].removeprefix("./")
        if len(match["reason"].split()) < MIN_REASON_WORDS:
            problems.append(
                f"the waiver for {target} needs a reason of at least {MIN_REASON_WORDS} words"
            )
            continue
        waivers[target] = match["reason"]
    return waivers, problems


# ---- routing --------------------------------------------------------------------------------


def load_areas(source: str) -> tuple[Area, ...]:
    """The ``AREAS`` of a ``tests/docs_map.py`` source, run as a throwaway module.

    The module is registered while it runs because :mod:`dataclasses` looks its module up.
    """
    name = "tests._docs_map_other"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, "tests/docs_map.py (other revision)", "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    fields = dataclasses.fields(Area)
    return tuple(
        Area(**{f.name: getattr(a, f.name, f.default) for f in fields})
        for a in module.__dict__["AREAS"]
    )


def matches(path: str, pattern: str) -> bool:
    """Whether ``path`` matches the glob ``pattern`` (``*`` crosses ``/``)."""
    return fnmatch.fnmatchcase(path, pattern)


def route(
    paths: Iterable[str], areas: Sequence[Area]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The gating and the listed-only pages the changed ``paths`` route to, with reasons."""
    gating: dict[str, list[str]] = {}
    listed: dict[str, list[str]] = {}
    for path in sorted(paths):
        for area in areas:
            if any(matches(path, pattern) for pattern in area.code):
                reason = f"{path} ({area.name})"
                for page in area.pages:
                    if reason not in gating.setdefault(page, []):
                        gating[page].append(reason)
                for page in area.see_also:
                    if reason not in listed.setdefault(page, []):
                        listed[page].append(reason)
    for page in gating:
        listed.pop(page, None)
    return gating, listed


def guards_for(page: str, areas: Sequence[Area]) -> list[str]:
    """The tests that already tie part of ``page`` to the code."""
    return sorted({g for a in areas if page in a.pages or page in a.see_also for g in a.guards})


# ---- the public surface ---------------------------------------------------------------------


@dataclass
class Surface:
    """What a reader could quote from one module, or from the whole package.

    Parameters
    ----------
    names : set of str
        Exported names, public functions and classes, public members of public classes,
        public parameters, and the string options of ``Literal`` annotations.
    defaults : dict
        ``(owner, parameter)`` to the source text of its default, where the owner is the
        function, ``Class.method`` or the class of a field.
    """

    names: set[str] = field(default_factory=set)
    defaults: dict[tuple[str, str], str] = field(default_factory=dict)

    def __ior__(self, other: Surface) -> Surface:
        self.names |= other.names
        self.defaults.update(other.defaults)
        return self


def _literals(annotation: ast.expr | None) -> set[str]:
    found: set[str] = set()
    if annotation is None:
        return found
    for node in ast.walk(annotation):
        if isinstance(node, ast.Subscript) and ast.unparse(node.value).split(".")[-1] == "Literal":
            for item in ast.walk(node.slice):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    found.add(item.value)
    return found


def exports(source: str | None) -> set[str]:
    """The names a module's ``__all__`` lists, in every form the package writes it.

    That is ``__all__ = [...]`` (or a tuple), ``__all__ += [...]`` and
    ``__all__.extend([...])``, with string elements.
    """
    names: set[str] = set()
    if not source:
        return names
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names

    def strings(node: ast.expr | None) -> set[str]:
        if isinstance(node, (ast.List, ast.Tuple)):
            return {
                e.value
                for e in node.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            }
        return set()

    def is_all(node: ast.expr) -> bool:
        return isinstance(node, ast.Name) and node.id == "__all__"

    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(is_all(t) for t in node.targets)) or (
            isinstance(node, (ast.AugAssign, ast.AnnAssign)) and is_all(node.target)
        ):
            names |= strings(node.value)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and is_all(node.value.func.value)
            and node.value.func.attr == "extend"
            and node.value.args
        ):
            names |= strings(node.value.args[0])
    return names


def surface(source: str | None, only: set[str] | None = None) -> Surface:
    """The public surface of one module's source; empty when it is missing or does not parse.

    With ``only``, keep just the top-level names in it, and the members, parameters and options
    of those: the part of the module a published namespace re-exports.
    """
    result = Surface()
    if not source:
        return result
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    def function(node: ast.FunctionDef | ast.AsyncFunctionDef, owner: str) -> None:
        a = node.args
        positional = [*a.posonlyargs, *a.args]
        pairs = list(zip(positional[len(positional) - len(a.defaults) :], a.defaults, strict=True))
        pairs += [
            (arg, d) for arg, d in zip(a.kwonlyargs, a.kw_defaults, strict=True) if d is not None
        ]
        for arg in (*positional, *a.kwonlyargs):
            if arg.arg in {"self", "cls"} or arg.arg.startswith("_"):
                continue
            result.names.add(arg.arg)
            result.names |= _literals(arg.annotation)
        for arg, default in pairs:
            if not arg.arg.startswith("_") and arg.arg not in {"self", "cls"}:
                result.defaults[(owner, arg.arg)] = ast.unparse(default)

    result.names |= {n for n in exports(source) if only is None or n in only}
    for node in tree.body:
        # A module-level alias such as ``MSMLink = Literal["identity", "logit"]`` names the
        # options of every signature that uses it.
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            result.names |= _literals(node.value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_") and (only is None or node.name in only):
                result.names.add(node.name)
                function(node, node.name)
        elif (
            isinstance(node, ast.ClassDef)
            and not node.name.startswith("_")
            and (only is None or node.name in only)
        ):
            result.names.add(node.name)
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    name = item.target.id
                    if name.startswith("_"):
                        continue
                    result.names.add(name)
                    result.names |= _literals(item.annotation)
                    if item.value is not None:
                        result.defaults[(node.name, name)] = ast.unparse(item.value)
                elif isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            result.names.add(target.id)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "__init__":
                        function(item, node.name)
                    elif not item.name.startswith("_"):
                        result.names.add(item.name)
                        function(item, f"{node.name}.{item.name}")
    result.names = {n for n in result.names if len(n) >= MIN_NAME_LENGTH}
    return result


def package_surface(read: Mapping[str, str | None]) -> Surface:
    """The union of the surfaces of the modules in ``read`` (path to source)."""
    total = Surface()
    for source in read.values():
        total |= surface(source)
    return total


def published(sources: Mapping[str, str | None]) -> set[str]:
    """The names the ``__all__`` of the modules of :data:`tests.docs_map.PUBLISHED` list."""
    names: set[str] = set()
    for path in PUBLISHED:
        names |= exports(sources.get(path))
    return names


def package_sources(revision: str | None, root: Path = ROOT) -> dict[str, str | None]:
    """Every ``src/cleverly`` module at ``revision``, or in the working tree when ``None``."""
    if revision is None:
        return {
            p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in (root / "src" / "cleverly").rglob("*.py")
        }
    listing = git("ls-tree", "-r", "--name-only", revision, "--", "src/cleverly", root=root)
    return blobs(revision, [p for p in listing.splitlines() if p.endswith(".py")], root)


def blobs(revision: str, paths: Sequence[str], root: Path = ROOT) -> dict[str, str | None]:
    """The text of many files at ``revision`` through one ``git cat-file --batch``."""
    if not paths:
        return {}
    done = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="".join(f"{revision}:{p}\n" for p in paths).encode(),
        capture_output=True,
        check=True,
    )
    out: dict[str, str | None] = {}
    data, at = done.stdout, 0
    for path in paths:
        end = data.index(b"\n", at)
        header = data[at:end].split()
        at = end + 1
        if len(header) < 3 or header[1] == b"missing":
            out[path] = None
            continue
        size = int(header[2])
        out[path] = data[at : at + size].decode("utf-8", errors="replace")
        at += size + 1
    return out


# ---- the documentation ----------------------------------------------------------------------


def doc_texts(change: Change, root: Path = ROOT) -> tuple[list[str], dict[str, str]]:
    """The documentation sources on the head side of ``change``, and their text."""
    if change.worktree or change.head == "HEAD":
        pages = doc_pages(root)
        return pages, {p: (root / p).read_text(encoding="utf-8", errors="replace") for p in pages}
    listing = git("ls-tree", "-r", "--name-only", change.head, root=root).splitlines()
    pages = sorted(
        p
        for p in listing
        if (p == "README.md" or (p.startswith("docs/") and p.endswith((".md", ".rst", ".ipynb"))))
        and not EXCLUDED_PARTS.intersection(p.split("/"))
        and not HISTORY_PAGES.match(p)
        and not GENERATED_PAGES.match(p)
    )
    return pages, {p: text or "" for p, text in blobs(change.head, pages, root).items()}


def doc_pages(root: Path = ROOT) -> list[str]:
    """Every hand-written documentation source, history and generated study pages excluded."""
    found = {"README.md"} if (root / "README.md").exists() else set()
    for suffix in ("md", "rst", "ipynb"):
        for path in (root / "docs").glob(f"**/*.{suffix}"):
            rel = path.relative_to(root)
            if not EXCLUDED_PARTS.intersection(rel.parts):
                found.add(rel.as_posix())
    return sorted(p for p in found if not HISTORY_PAGES.match(p) and not GENERATED_PAGES.match(p))


#: A code span: backticks paired left to right, so the prose between two spans is not code.
SPAN = re.compile(r"`[^`\n]*`")


def quoted(names: Iterable[str], texts: Mapping[str, str]) -> dict[str, list[tuple[int, str]]]:
    """Per page, each ``(line, name)`` where a name sits in a code span or reads ``cleverly.x``."""
    names = list(names)
    hits: dict[str, list[tuple[int, str]]] = {}
    if not names:
        return hits
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    word = re.compile(rf"(?<![\w-])({alternatives})(?![\w-])")
    dotted = re.compile(rf"\bcleverly(?:\.\w+)*\.({alternatives})\b")
    for page, text in texts.items():
        for number, line in enumerate(text.splitlines(), start=1):
            found = [m.group(1) for span in SPAN.findall(line) for m in word.finditer(span)]
            outside = SPAN.sub("", line)
            found += [m.group(1) for m in dotted.finditer(outside)]
            for name in found:
                hits.setdefault(page, []).append((number, name))
    return hits


def mentioned(names: Iterable[str], texts: Mapping[str, str]) -> set[str]:
    """The names that appear as a whole word anywhere in the documentation."""
    names = set(names)
    if not names:
        return set()
    alternatives = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    pattern = re.compile(rf"(?<![\w-])({alternatives})(?![\w-])")
    return {
        m.group(1)
        for page, text in texts.items()
        if page not in PLANNED_PAGES
        for m in pattern.finditer(text)
    }


# ---- the verdict ----------------------------------------------------------------------------


@dataclass
class Finding:
    """One thing the change must answer.

    Parameters
    ----------
    target : str
        A page path, or a name for an undocumented addition.
    kind : str
        ``routed``, ``removed``, ``default`` or ``added``.
    status : str
        ``ok`` (edited, or no longer quoted, or mentioned), ``waived`` or ``MISSING``.
    reasons : list of str
        What produced the finding.
    guards : list of str
        Tests that already tie part of the page to the code.
    """

    target: str
    kind: str
    status: str
    reasons: list[str]
    guards: list[str] = field(default_factory=list)


@dataclass
class Report:
    """The verdict on one change.

    Parameters
    ----------
    findings : list of Finding
        Every routed page, quoted removed name, old default and addition.
    listed : dict
        The ``see_also`` pages, with the reasons; never a failure.
    problems : list of str
        Malformed, unknown and stale waivers.
    map_changes : list of str
        Code globs or pages this change takes out of the map.
    """

    findings: list[Finding] = field(default_factory=list)
    listed: dict[str, list[str]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    map_changes: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[Finding]:
        """The findings that are neither satisfied nor waived."""
        return [f for f in self.findings if f.status == "MISSING"]

    @property
    def ok(self) -> bool:
        """Whether the change passes."""
        return not self.missing and not self.problems


def evaluate(
    *,
    gating: Mapping[str, list[str]],
    listed: Mapping[str, list[str]],
    edited: Iterable[str],
    stale_names: Mapping[str, list[str]],
    stale_defaults: Mapping[str, list[str]],
    additions: Mapping[str, str],
    documented: Iterable[str],
    waivers: Mapping[str, str],
    problems: Sequence[str] = (),
    pages: Iterable[str] = (),
    areas: Sequence[Area] = AREAS,
) -> Report:
    """Combine what the change touched, what it removed and added, and the waivers."""
    edited = set(edited)
    documented = set(documented)
    report = Report(listed=dict(listed), problems=list(problems))
    blanket = "all" in waivers

    def status(satisfied: bool, target: str, *, blanket_ok: bool) -> str:
        if satisfied:
            return "ok"
        if target in waivers or (blanket_ok and blanket):
            return "waived"
        return "MISSING"

    for page in sorted(gating):
        report.findings.append(
            Finding(
                page,
                "routed",
                status(page in edited, page, blanket_ok=True),
                list(gating[page]),
                guards_for(page, areas),
            )
        )
    for kind, stale in (("removed", stale_names), ("default", stale_defaults)):
        for page in sorted(stale):
            report.findings.append(
                Finding(
                    page,
                    kind,
                    status(False, page, blanket_ok=False),
                    list(stale[page]),
                    guards_for(page, areas),
                )
            )
    for name in sorted(set(additions) - documented):
        report.findings.append(
            Finding(
                name,
                "added",
                status(False, name, blanket_ok=False),
                [f"new public name in {additions[name]}, mentioned on no page"],
            )
        )
    targets = {f.target for f in report.findings}
    known_pages = set(pages)
    for target in waivers:
        if target == "all":
            if not gating:
                report.problems.append("the waiver for all covers nothing: no page is routed")
        elif "/" in target or target.endswith((".md", ".ipynb", ".rst")):
            if known_pages and target not in known_pages:
                report.problems.append(f"the waiver names {target}, which is not a doc page")
            elif target not in targets:
                report.problems.append(f"the waiver for {target} is stale: no finding names it")
        elif target not in targets:
            report.problems.append(f"the waiver for {target} is stale: no finding names it")
    return report


def map_changes(base: Sequence[Area], head: Sequence[Area]) -> list[str]:
    """Code globs and pages present in the base map and missing from the head map."""
    out: list[str] = []
    base_code = {g for a in base for g in a.code}
    head_code = {g for a in head for g in a.code}
    out += [f"code glob {g} left the map" for g in sorted(base_code - head_code)]
    base_pages = {(g, p) for a in base for g in a.code for p in a.pages}
    head_pages = {(g, p) for a in head for g in a.code for p in a.pages}
    for g, p in sorted(base_pages - head_pages):
        out.append(f"{p} no longer gates on {g}")
    return out


def render(report: Report, waivers: Mapping[str, str], *, ci: bool) -> str:
    """The report as Markdown.

    In CI no waiver reason is echoed, so a body line cannot become a workflow command; the
    step summary, which GitHub does not parse for commands, gets the reasons. Table cells
    escape ``|``.
    """

    def cell(text: str) -> str:
        return text.replace("|", "\\|")

    out = ["## Docs impact", ""]
    if report.map_changes:
        out.append("**This change narrows the map that judges it** (the base map still applies):")
        out += [f"- {line}" for line in report.map_changes]
        out.append("")
    if not report.findings and not report.problems:
        out.append("No mapped code changed, and no removed or added public name needs a page.")
    if report.findings:
        out += ["| target | why | status | guarded in part by |", "| --- | --- | --- | --- |"]
        for f in report.findings:
            why = "; ".join(f.reasons)
            status = f.status
            if f.status == "waived" and not ci:
                status += f": {waivers.get(f.target) or waivers.get('all')}"
            guards = ", ".join(f.guards)
            out.append(
                f"| `{f.target}` | {cell(f.kind + ': ' + why)} | {cell(status)} | {guards} |"
            )
        out.append("")
    if report.listed:
        out.append("Also worth a look (never fails):")
        out += [f"- `{page}`: {'; '.join(why)}" for page, why in sorted(report.listed.items())]
        out.append("")
    out += [f"Problem: {problem}" for problem in report.problems]
    if report.missing:
        out += [
            "",
            "Read each MISSING page against the change and fix what it made stale. For a page or"
            " a name that needs nothing, add this line to the pull request body with the reason:",
            "",
        ]
        out += [f"    Docs-waiver: {f.target}: <reason>" for f in report.missing]
    out += ["", "PASS" if report.ok else "FAIL"]
    return "\n".join(out) + "\n"


def pr_body(args: argparse.Namespace) -> str:
    """The pull request body: the file given, else the event payload, else ``PR_BODY``."""
    if args.body_file is not None:
        return args.body_file.read_text(encoding="utf-8")
    event = os.environ.get("GITHUB_EVENT_PATH")
    if event and Path(event).exists():
        payload = json.loads(Path(event).read_text(encoding="utf-8"))
        return (payload.get("pull_request") or {}).get("body") or ""
    return os.environ.get("PR_BODY", "")


def check(base: str, body: str, *, worktree: bool, head: str = "HEAD", root: Path = ROOT) -> Report:
    """The whole check, from the git state and a pull request body."""
    change = changed(base, worktree=worktree, head=head, root=root)
    head_map = change.head_text("tests/docs_map.py", root)
    base_map = show(change.merge_base, "tests/docs_map.py", root)
    head_areas = load_areas(head_map) if head_map else AREAS
    base_areas = load_areas(base_map) if base_map else ()
    areas = (*base_areas, *head_areas)
    gating, listed = route(change.paths, areas)

    modules = sorted(p for p in change.paths if p.startswith("src/cleverly/") and p.endswith(".py"))
    lost: dict[str, str] = {}
    new: dict[str, str] = {}
    old_defaults: dict[tuple[str, str], tuple[str, str]] = {}
    if modules:
        head_sources = package_sources(None if worktree else head, root)
        base_sources = package_sources(change.merge_base, root)
        exported = published(head_sources)
        base_exported = published(base_sources)
        for path in modules:
            before_text = base_sources.get(path)
            after_text = change.head_text(path, root)
            before, after = surface(before_text), surface(after_text)
            # Only what the base branch published can be quoted as API: a helper of a private
            # module, or one left out of ``__all__``, is not.
            for name in surface(before_text, base_exported).names - after.names:
                lost.setdefault(name, path)
            for name in surface(after_text, exported).names - before.names:
                new.setdefault(name, path)
            for key, value in before.defaults.items():
                if key in after.defaults and after.defaults[key] != value:
                    old_defaults[key] = (value, path)
        head_all = package_surface(head_sources)
        lost = {n: p for n, p in lost.items() if n not in head_all.names}
        base_all = package_surface(base_sources)
        new = {n: p for n, p in new.items() if n not in base_all.names}

    pages, texts = doc_texts(change, root)
    stale_names: dict[str, list[str]] = {}
    for page, hits in quoted(lost, texts).items():
        stale_names[page] = [
            f"line {n} quotes `{name}`, removed from {lost[name]}" for n, name in hits
        ]
    stale_defaults: dict[str, list[str]] = {}
    for (owner, param), (value, _path) in sorted(old_defaults.items()):
        forms = {f"{param}={value}", f"{param}={value.replace(chr(39), chr(34))}"}
        for page, text in texts.items():
            for number, line in enumerate(text.splitlines(), start=1):
                spans = re.findall(r"`[^`\n]*`", line)
                if not any(form in span for span in spans for form in forms):
                    continue
                # A quoted ``kw=old`` is often an explicit choice that stays valid; only a line
                # that calls it the default is a claim the change made false.
                where = stale_defaults if "default" in line.lower() else listed
                reason = f"line {number} quotes `{param}={value}`, the old default of {owner}"
                if reason not in where.setdefault(page, []):
                    where[page].append(reason)
    documented = mentioned(new, texts)
    waivers, problems = parse_waivers(body)
    report = evaluate(
        gating=gating,
        listed=listed,
        edited=change.edited,
        stale_names=stale_names,
        stale_defaults=stale_defaults,
        additions=new,
        documented=documented,
        waivers=waivers,
        problems=problems,
        pages=pages,
        areas=areas,
    )
    report.map_changes = map_changes(base_areas, head_areas) if base_areas else []
    return report


def main(argv: Sequence[str] | None = None, root: Path = ROOT) -> int:
    """Run the check; return 0 on a pass and 1 on a failure."""
    parser = argparse.ArgumentParser(
        prog="python -m tests.docs_impact",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default="origin/main", help="the branch the change merges into")
    parser.add_argument(
        "--worktree", action="store_true", help="include uncommitted and untracked files"
    )
    parser.add_argument(
        "--body-file",
        type=Path,
        help="the pull request body (default: the GitHub event, else PR_BODY)",
    )
    args = parser.parse_args(argv)
    ci = os.environ.get("GITHUB_ACTIONS") == "true"
    body = pr_body(args)
    report = check(args.base, body, worktree=args.worktree, root=root)
    waivers, _ = parse_waivers(body)
    if ci:
        token = secrets.token_hex(16)
        sys.stdout.write(f"::stop-commands::{token}\n")
    sys.stdout.write(render(report, waivers, ci=ci))
    if ci:
        sys.stdout.write(f"::{token}::\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(render(report, waivers, ci=False))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
