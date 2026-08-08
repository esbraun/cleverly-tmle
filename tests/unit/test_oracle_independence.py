"""The discrete-law oracles must not import the library they are used to check.

``tests/discrete_law*.py`` write the identification formula out longhand and obtain
the efficient influence function by complex-step differentiation of the contamination
path.  Their value is that they are an *independent* re-derivation: when
``test_influence_gateaux.py`` agrees with them to 1e-12, that is evidence about the
library rather than a tautology.

That independence is a property of the source, not of anyone's good intentions, so it
is asserted here.  The moment an oracle imports ``cleverly`` -- even for something
that looks harmless, like reusing ``expit`` or a bounds helper -- the agreement stops
being evidence, because a sign error in the shared helper would move both sides
equally and the test would still pass.

This is deliberately *not* a rule against a commit touching both the oracles and
``src/``: adding a new estimand together with the oracle that validates it is exactly
the workflow the package wants.  What must never happen is the oracle depending on
the code under test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ORACLES = sorted((Path(__file__).resolve().parents[1]).glob("discrete_law*.py"))


def test_oracle_modules_exist() -> None:
    """Guard against the glob silently matching nothing after a rename."""
    names = {path.name for path in ORACLES}
    assert {"discrete_law.py", "discrete_law_mar.py", "discrete_law_cde.py"} <= names


@pytest.mark.parametrize("path", ORACLES, ids=lambda p: p.name)
def test_oracle_does_not_import_cleverly(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            offenders += [alias.name for alias in node.names if _is_cleverly(alias.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and _is_cleverly(node.module):
            offenders.append(node.module)

    assert not offenders, (
        f"{path.name} imports {sorted(set(offenders))} from the library it is meant to "
        "check independently. The oracle must re-derive the functional and its influence "
        "curve from scratch; sharing code with src/ turns the Gateaux tests into a "
        "tautology."
    )


def _is_cleverly(module: str) -> bool:
    return module == "cleverly" or module.startswith("cleverly.")
