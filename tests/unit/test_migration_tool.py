"""The optional migration audit finds old call shapes without rewriting intent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.conftest import ROOT

SCRIPT = ROOT / "scripts" / "migrate_public_api.py"


def run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_a_clean_typed_workflow_passes(tmp_path: Path) -> None:
    source = tmp_path / "new.py"
    source.write_text(
        "from cleverly import ATE, CausalStudy, PointTreatment\n"
        "study = CausalStudy(frame, design=PointTreatment(outcome='Y', treatment='A', "
        "adjustment=('W',)))\n"
        "result = study.estimate(ATE())\n",
        encoding="utf-8",
    )
    completed = run(source)
    assert completed.returncode == 0
    assert "No former root API calls" in completed.stdout


def test_old_import_fit_roles_and_single_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "old.py"
    source.write_text(
        "from cleverly import TMLE\n"
        "fit = TMLE(estimands=('ate',)).fit(frame, outcome='Y', treatment='A', "
        "covariates=('W',))\n"
        "result = fit.single()\n",
        encoding="utf-8",
    )
    completed = run(source)
    assert completed.returncode == 1
    assert "root import TMLE was removed" in completed.stdout
    assert "move estimator fit role(s)" in completed.stdout
    assert ".single() was removed" in completed.stdout
    assert "no files were changed" in completed.stdout
