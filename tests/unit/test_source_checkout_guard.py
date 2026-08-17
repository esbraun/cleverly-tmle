"""The guard that refuses a run whose source came from another checkout.

:func:`tests.conftest._check_source_matches_checkout` is a hook, not a test, so nothing
else in the suite can observe it working -- and a guard that only ever stays silent is
indistinguishable from one that is wired to nothing.  These are its witnesses: the
mismatch it exists for must raise, and the two shapes it deliberately permits must not.

The failure it catches is quiet by construction.  Every ``git worktree`` of this
repository shares one editable install, which points at the tree it was installed from,
so ``pytest`` inside a worktree pairs that worktree's tests with another branch's source.
Nothing announces the pairing; it surfaces as ordinary assertion failures in whichever
tests arrived with the branch, and the sole tell is a ``../..`` prefix on the traceback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import cleverly
from tests.conftest import _check_source_matches_checkout


def test_the_real_run_is_accepted() -> None:
    """The control: the guard is silent for the configuration the suite runs under.

    Without this, a guard that raised unconditionally and a guard that raised correctly
    would look the same from the mismatch test alone -- and every session would be
    failing for the wrong reason.
    """
    _check_source_matches_checkout()


def test_a_sibling_checkout_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Same layout, different tree: the case with no other symptom."""
    other = tmp_path / "cleverly-tmle-other" / "src" / "cleverly"
    monkeypatch.setattr(cleverly, "__file__", str(other / "__init__.py"))
    with pytest.raises(pytest.UsageError) as error:
        _check_source_matches_checkout()
    message = str(error.value)
    assert str(other) in message, "the refusal has to name the tree that was imported"
    assert "PYTHONPATH" in message, "and the one-run workaround, or it is only a complaint"


def test_an_installed_copy_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A wheel in ``site-packages`` is a deliberate configuration, not the mistake.

    The guard keys on the ``src/`` parent that only a checkout has, so this is the edge
    that says it discriminates rather than simply refusing anything unfamiliar.
    """
    installed = tmp_path / "venv" / "Lib" / "site-packages" / "cleverly"
    monkeypatch.setattr(cleverly, "__file__", str(installed / "__init__.py"))
    _check_source_matches_checkout()
