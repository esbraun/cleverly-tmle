"""The docs-impact check finds what a change made stale, and honours only real waivers.

:mod:`tests.docs_impact` runs on every pull request. These tests drive it on a scratch git
repository with a two-area map, so each finding is produced by a real diff: a routed page, a
removed name still quoted, a default a page still calls the default, and an addition no page
mentions. Each has a control where the finding must not fire, because a check that always fires
teaches people to waive it and a check that never fires checks nothing.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tests import docs_impact
from tests.docs_impact import parse_waivers, surface

MAP = """
from dataclasses import dataclass


@dataclass(frozen=True)
class Area:
    name: str
    code: tuple
    pages: tuple
    guards: tuple = ()
    see_also: tuple = ()


AREAS = (
    Area("tools", ("src/cleverly/tools.py",), ("docs/tools.md",), (), ("docs/extra.md",)),
    Area("other", ("src/cleverly/other.py",), ("docs/other.md",)),
)
"""

TOOLS = """
__all__ = ["fit", "Settings"]


def fit(data, grid=3, mode="fast"):
    return data


class Settings:
    alpha: float = 0.05
    method: str = "auto"

    def summary(self):
        return ""
"""


def run(root: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True, env=env
    ).stdout


def write(root: Path, path: str, text: str) -> None:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A repository on ``main`` with two mapped modules and their pages, branched to ``work``."""
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "-q", str(tmp_path)], check=True
    )
    write(tmp_path, "tests/docs_map.py", MAP)
    write(tmp_path, "src/cleverly/tools.py", TOOLS)
    write(tmp_path, "src/cleverly/other.py", "def other_thing(x):\n    return x\n")
    write(
        tmp_path,
        "src/cleverly/__init__.py",
        'from .tools import fit, Settings\n__all__ = ["fit", "Settings"]\n',
    )
    write(
        tmp_path,
        "docs/tools.md",
        '# Tools\n\nCall `fit(data, grid=3)`.\n\nThe default is `mode="fast"`.\n',
    )
    write(tmp_path, "docs/other.md", "# Other\n\nRead `Settings.summary()` and `other_thing`.\n")
    write(tmp_path, "docs/extra.md", "# Extra\n")
    write(tmp_path, "README.md", "# Readme\n")
    run(tmp_path, "add", "-A")
    run(tmp_path, "commit", "-qm", "base")
    run(tmp_path, "checkout", "-qb", "work")
    monkeypatch.setattr(docs_impact, "PUBLISHED", ("src/cleverly/__init__.py",))
    return tmp_path


def commit(root: Path, message: str = "change") -> None:
    run(root, "add", "-A")
    run(root, "commit", "-qm", message)


def check(root: Path, body: str = "") -> docs_impact.Report:
    return docs_impact.check("main", body, worktree=False, root=root)


def statuses(report: docs_impact.Report) -> dict[tuple[str, str], str]:
    return {(f.kind, f.target): f.status for f in report.findings}


# ---- waivers ---------------------------------------------------------------------------------


def test_waivers_parse_in_their_forms() -> None:
    body = (
        "Intro\r\n"
        "- Docs-waiver: docs/a.md: only a private helper moved files\r\n"
        "* docs-waiver: `./docs/b.md`: the page never describes this code\n"
        "Docs-waiver: all: pure refactor with byte-identical output\n"
    )
    waivers, problems = parse_waivers(body)
    assert problems == []
    assert set(waivers) == {"docs/a.md", "docs/b.md", "all"}


def test_waivers_in_fences_and_comments_do_not_count() -> None:
    body = (
        "<!--\nDocs-waiver: docs/a.md: an example in the template comment\n-->\n"
        "```text\nDocs-waiver: docs/b.md: an example inside a code fence\n```\n"
    )
    assert parse_waivers(body) == ({}, [])


@pytest.mark.parametrize(
    "line",
    ["Docs-waiver: docs/a.md: too short", "Docs-waiver docs/a.md because reasons given here"],
)
def test_short_or_malformed_waivers_are_problems(line: str) -> None:
    waivers, problems = parse_waivers(line)
    assert waivers == {}
    assert len(problems) == 1


# ---- routing ---------------------------------------------------------------------------------


def test_a_routed_page_must_change_or_be_waived(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    commit(repo)
    report = check(repo)
    assert statuses(report) == {("routed", "docs/tools.md"): "MISSING"}
    assert "docs/extra.md" in report.listed
    assert not report.ok

    waived = check(repo, "Docs-waiver: docs/tools.md: a comment changes no behaviour at all")
    assert waived.ok

    write(
        repo,
        "docs/tools.md",
        '# Tools\n\nCall `fit(data, grid=3)`. New sentence.\nThe default is `mode="fast"`.\n',
    )
    commit(repo)
    assert check(repo).ok


def test_a_whitespace_touch_does_not_count(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    write(repo, "docs/tools.md", (repo / "docs/tools.md").read_text() + "\n\n")
    commit(repo)
    assert statuses(check(repo))[("routed", "docs/tools.md")] == "MISSING"


def test_unmapped_changes_need_nothing(repo: Path) -> None:
    write(repo, "notes.txt", "hello\n")
    commit(repo)
    report = check(repo)
    assert report.findings == [] and report.ok


def test_a_stale_or_unknown_waiver_fails(repo: Path) -> None:
    write(repo, "notes.txt", "hello\n")
    commit(repo)
    stale = check(repo, "Docs-waiver: docs/tools.md: nothing here routes to this page")
    assert not stale.ok and "stale" in stale.problems[0]
    unknown = check(repo, "Docs-waiver: docs/nope.md: this page does not exist anywhere")
    assert not unknown.ok and "not a doc page" in unknown.problems[0]


def test_a_rename_routes_the_old_path_too(repo: Path) -> None:
    run(repo, "mv", "src/cleverly/other.py", "src/cleverly/moved.py")
    commit(repo)
    assert ("routed", "docs/other.md") in statuses(check(repo))


def test_the_base_map_still_applies_when_the_change_narrows_it(repo: Path) -> None:
    write(
        repo,
        "tests/docs_map.py",
        MAP.replace('    Area("other", ("src/cleverly/other.py",), ("docs/other.md",)),\n', ""),
    )
    write(repo, "src/cleverly/other.py", "def other_thing(x):\n    return x + 1\n")
    commit(repo)
    report = check(repo)
    assert statuses(report)[("routed", "docs/other.md")] == "MISSING"
    assert any("src/cleverly/other.py" in line for line in report.map_changes)


# ---- removed names, defaults and additions ---------------------------------------------------


def test_a_removed_method_still_quoted_fails_until_the_page_drops_it(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS.replace("def summary", "def to_frame"))
    write(
        repo, "docs/other.md", "# Other\n\nRead `Settings.summary()` and `other_thing`. Edited.\n"
    )
    commit(repo)
    report = check(repo, "Docs-waiver: all: the rename is covered by the API page")
    assert statuses(report)[("removed", "docs/other.md")] == "MISSING"

    write(repo, "docs/other.md", "# Other\n\nRead `Settings.to_frame()` and `other_thing`.\n")
    commit(repo)
    assert ("removed", "docs/other.md") not in statuses(check(repo))


def test_a_removed_name_that_lives_on_elsewhere_is_not_reported(repo: Path) -> None:
    write(repo, "src/cleverly/other.py", "def other_thing(x, grid=1):\n    return x\n")
    write(repo, "src/cleverly/tools.py", TOOLS.replace("def fit(data, grid=3,", "def fit(data,"))
    commit(repo)
    kinds = {kind for kind, _ in statuses(check(repo))}
    assert "removed" not in kinds


def test_a_name_in_prose_without_backticks_is_not_a_quote() -> None:
    hits = docs_impact.quoted(
        {"summary"}, {"p.md": "A summary of `fit`.\n", "q.md": "`x.summary()`\n"}
    )
    assert set(hits) == {"q.md"}


def test_a_changed_default_fails_only_where_a_page_calls_it_the_default(repo: Path) -> None:
    write(
        repo,
        "src/cleverly/tools.py",
        TOOLS.replace('mode="fast"', 'mode="slow"').replace("grid=3", "grid=4"),
    )
    commit(repo)
    report = check(repo, "Docs-waiver: all: a blanket waiver never covers an old default")
    assert statuses(report)[("default", "docs/tools.md")] == "MISSING"
    # `grid=3` is quoted as an explicit call, not as the default: listed, never a failure.
    assert any("grid=3" in why for why in report.listed.get("docs/tools.md", []))


def test_an_undocumented_addition_fails_until_a_page_mentions_it(repo: Path) -> None:
    write(
        repo, "src/cleverly/tools.py", TOOLS.replace('mode="fast"', 'mode="fast", horizon_cap=None')
    )
    commit(repo)
    report = check(repo, "Docs-waiver: all: blanket waivers never cover new names")
    assert statuses(report)[("added", "horizon_cap")] == "MISSING"
    waived = check(repo, "Docs-waiver: horizon_cap: internal knob the harness sets, never by users")
    assert statuses(waived)[("added", "horizon_cap")] == "waived"

    write(
        repo,
        "docs/tools.md",
        (repo / "docs/tools.md").read_text() + "\nSet `horizon_cap` to stop early.\n",
    )
    commit(repo)
    assert ("added", "horizon_cap") not in statuses(check(repo))


def test_an_addition_outside_the_published_surface_is_not_reported(repo: Path) -> None:
    write(
        repo,
        "src/cleverly/other.py",
        "def other_thing(x):\n    return x\n\n\ndef helper_value(y):\n    return y\n",
    )
    commit(repo)
    assert "added" not in {kind for kind, _ in statuses(check(repo))}


def test_surface_reads_members_options_and_defaults() -> None:
    found = surface(
        "from typing import Literal\n"
        "class Alpha:\n"
        "    xval: Literal['one', 'two'] = 'one'\n"
        "    def run(self, *, level: int = 2): ...\n"
        "    @property\n"
        "    def width(self): ...\n"
        "def _private(z): ...\n"
    )
    assert {"Alpha", "run", "level", "width", "one", "two"} <= found.names
    assert "_private" not in found.names and "self" not in found.names
    assert found.defaults[("Alpha", "xval")] == "'one'"
    assert found.defaults[("Alpha.run", "level")] == "2"


def test_main_reports_and_exits(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    commit(repo)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.delenv("PR_BODY", raising=False)
    assert docs_impact.main(["--base", "main"], root=repo) == 1
    out = capsys.readouterr().out
    assert "Docs-waiver: docs/tools.md: <reason>" in out and out.rstrip().endswith("FAIL")
    body = repo / "body.md"
    body.write_text("Docs-waiver: docs/tools.md: a comment changes no behaviour at all\n")
    assert docs_impact.main(["--base", "main", "--body-file", str(body)], root=repo) == 0


def test_ci_output_never_echoes_the_body(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    commit(repo)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(repo / "summary.md"))
    monkeypatch.setenv("PR_BODY", "Docs-waiver: docs/tools.md: ::error::injected words that run")
    assert docs_impact.main(["--base", "main"], root=repo) == 0
    out = capsys.readouterr().out
    assert "injected" not in out and out.startswith("::stop-commands::")
    assert "injected" in (repo / "summary.md").read_text()


# ---- review controls -------------------------------------------------------------------------


def test_prose_between_two_code_spans_is_not_a_quote() -> None:
    hits = docs_impact.quoted({"starts"}, {"p.md": "The `0.2.N` series starts at `0.2.0`.\n"})
    assert hits == {}


def test_exports_reads_every_form_of_all() -> None:
    source = '__all__ = ["one_name"]\n__all__ += ["two_name"]\n__all__.extend(("three_name",))\n'
    assert docs_impact.exports(source) == {"one_name", "two_name", "three_name"}


def test_a_blanket_waiver_satisfies_routed_pages(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    commit(repo)
    report = check(repo, "Docs-waiver: all: a comment changes no behaviour at all")
    assert statuses(report) == {("routed", "docs/tools.md"): "waived"} and report.ok


def test_a_blanket_waiver_with_nothing_routed_is_a_problem(repo: Path) -> None:
    write(repo, "notes.txt", "hello\n")
    commit(repo)
    assert not check(repo, "Docs-waiver: all: nothing at all is routed here").ok


def test_an_indentation_only_page_edit_does_not_count(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# a comment\n")
    page = (repo / "docs/tools.md").read_text().replace("Call", "  Call")
    write(repo, "docs/tools.md", page)
    commit(repo)
    assert statuses(check(repo))[("routed", "docs/tools.md")] == "MISSING"


def test_a_removed_parameter_still_quoted_is_found(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS.replace("def fit(data, grid=3,", "def fit(data,"))
    commit(repo)
    report = check(repo, "Docs-waiver: docs/tools.md: the waiver covers this page by name")
    assert ("removed", "docs/tools.md") in statuses(report)


def test_a_removed_literal_option_of_an_alias_is_found(repo: Path) -> None:
    alias = 'from typing import Literal\nMode = Literal["fast", "careful"]\n'
    write(
        repo, "src/cleverly/tools.py", alias + TOOLS.replace('mode="fast"', 'mode: Mode = "fast"')
    )
    write(repo, "docs/other.md", '# Other\n\nUse `mode="careful"`.\n')
    commit(repo, "add the alias")
    run(repo, "branch", "-f", "main", "HEAD")
    write(
        repo,
        "src/cleverly/tools.py",
        (repo / "src/cleverly/tools.py").read_text().replace(', "careful"', ""),
    )
    commit(repo)
    report = check(repo, "Docs-waiver: docs/tools.md: the alias change needs no edit here")
    assert ("removed", "docs/other.md") in statuses(report)


def test_a_removed_helper_outside_the_published_surface_is_not_reported(repo: Path) -> None:
    write(
        repo,
        "src/cleverly/other.py",
        "def other_thing(x):\n    return x\n\n\ndef private_helper(y):\n    return y\n",
    )
    write(
        repo,
        "docs/other.md",
        "# Other\n\nRead `Settings.summary()`, `other_thing` and `private_helper`.\n",
    )
    commit(repo, "add helper")
    run(repo, "branch", "-f", "main", "HEAD")
    write(repo, "src/cleverly/other.py", "def other_thing(x):\n    return x\n")
    commit(repo)
    assert "removed" not in {kind for kind, _ in statuses(check(repo))}


def test_a_rename_reports_no_removed_name(repo: Path) -> None:
    run(repo, "mv", "src/cleverly/other.py", "src/cleverly/moved.py")
    commit(repo)
    assert "removed" not in {kind for kind, _ in statuses(check(repo))}


def test_the_worktree_counts_uncommitted_and_untracked_files(repo: Path) -> None:
    write(repo, "src/cleverly/tools.py", TOOLS + "\n# uncommitted\n")
    write(repo, "src/cleverly/brand_new.py", "x = 1\n")
    report = docs_impact.check("main", "", worktree=True, root=repo)
    assert statuses(report) == {("routed", "docs/tools.md"): "MISSING"}
    change = docs_impact.changed("main", worktree=True, root=repo)
    assert "src/cleverly/brand_new.py" in change.paths


def test_a_pipe_in_a_waiver_reason_is_escaped_in_the_table() -> None:
    report = docs_impact.Report(
        findings=[docs_impact.Finding("docs/a.md", "routed", "waived", ["x"])]
    )
    text = docs_impact.render(report, {"docs/a.md": "one | two | three words"}, ci=False)
    row = next(line for line in text.splitlines() if line.startswith("| `docs/a.md`"))
    assert "one \\| two" in row
