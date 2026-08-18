"""Audit Python source for calls removed by the causal-question public API.

This is intentionally a reporter, not a semantic codemod.  Rewriting ``estimands=("ate",)``
without reading the analysis could change the question, and moving a role such as ``delta=``
requires constructing the correct design object.  The diagnostics identify the straightforward
locations and point to the migration guide; a clean run is suitable as a temporary local gate.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

LEGACY_ROOT = {"TMLE", "LTMLE", "CTMLE", "DRTMLE", "tmle", "ltmle"}
FIT_ROLES = {
    "outcome",
    "treatment",
    "covariates",
    "baseline",
    "time_varying",
    "censoring",
    "delta",
    "intermediate",
    "weights",
    "weights_type",
    "weights_estimated",
    "id",
    "strata",
    "family",
    "treatment_kind",
}
SKIP_PARTS = {".git", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".venv"}


@dataclass(frozen=True, order=True)
class Finding:
    path: Path
    line: int
    column: int
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.message}"


class MigrationVisitor(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.findings: list[Finding] = []
        self.legacy_names: set[str] = set()
        self.cleverly_names: set[str] = {"cleverly"}

    def add(self, node: ast.AST, message: str) -> None:
        self.findings.append(
            Finding(
                self.path,
                getattr(node, "lineno", 1),
                getattr(node, "col_offset", 0) + 1,
                message,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "cleverly":
                self.cleverly_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "cleverly":
            for alias in node.names:
                if alias.name in LEGACY_ROOT:
                    local = alias.asname or alias.name
                    self.legacy_names.add(local)
                    self.add(
                        node,
                        f"root import {alias.name} was removed; construct CausalStudy with a "
                        "typed design and estimand",
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            node.attr in LEGACY_ROOT
            and isinstance(node.value, ast.Name)
            and node.value.id in self.cleverly_names
        ):
            self.add(
                node,
                f"cleverly.{node.attr} was removed; use the causal-question workflow",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.legacy_names:
            self.add(
                node,
                f"{node.func.id}(...) is a former root entry point; move roles to a design, "
                "the question to a typed estimand, and settings to a method",
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "single":
            self.add(
                node,
                ".single() was removed from ordinary public fits; the fit already returns the "
                "CausalResult (do not replace this call with .estimate)",
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
            roles = sorted(
                keyword.arg
                for keyword in node.keywords
                if keyword.arg is not None and keyword.arg in FIT_ROLES
            )
            if roles:
                self.add(
                    node,
                    "move estimator fit role(s) to PointTreatment/LongitudinalTreatment: "
                    + ", ".join(roles),
                )
        self.generic_visit(node)


def python_files(arguments: Iterable[str]) -> list[Path]:
    files: set[Path] = set()
    for raw in arguments:
        path = Path(raw)
        if path.is_file():
            if path.suffix == ".py":
                files.add(path)
            continue
        if path.is_dir():
            files.update(
                candidate
                for candidate in path.rglob("*.py")
                if not any(part in SKIP_PARTS for part in candidate.parts)
            )
            continue
        raise FileNotFoundError(f"migration path does not exist: {path}")
    return sorted(files)


def audit(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        return [
            Finding(
                path,
                error.lineno or 1,
                error.offset or 1,
                f"cannot audit invalid Python: {error.msg}",
            )
        ]
    visitor = MigrationVisitor(path)
    visitor.visit(tree)
    return visitor.findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report former cleverly root API calls that need causal-question migration."
    )
    parser.add_argument("paths", nargs="+", help="Python files or directories to audit")
    args = parser.parse_args()

    findings = [finding for path in python_files(args.paths) for finding in audit(path)]
    for finding in sorted(findings):
        print(finding.render())
    if findings:
        print(
            f"\n{len(findings)} migration finding(s). See docs/migration.md; no files were changed."
        )
        return 1
    print("No former root API calls found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
