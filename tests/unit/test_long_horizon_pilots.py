"""The part-indexed pilot driver: a fixed part list, resumable parts, a merge that checks seeds."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd
import pytest

from tests import long_horizon_design as D
from tests import long_horizon_pilots as P
from tests.discrete_law_longitudinal_horizon import L_R, Law


def test_the_part_list_is_fixed_and_every_part_has_its_own_file() -> None:
    every = P.parts()
    assert every == P.parts.__wrapped__()
    assert len({part.ident for part in every}) == len(every)
    assert len({part.filename for part in every}) == len(every)
    families = [part.family for part in every]
    # Exact scans first, then the sample streams in the declared stream order.
    assert families.index("mc") > max(i for i, f in enumerate(families) if f == "patterns")
    for stream in D.streams():
        blocks = [p.detail for p in every if p.name == stream.ident]
        assert blocks[0][0] == 0
        assert blocks[-1][1] == stream.replicates
        assert all(a[1] == b[0] for a, b in itertools.pairwise(blocks))


def test_split_and_index_ranges(capsys: pytest.CaptureFixture[str]) -> None:
    P.main(["split", "--share", "0.2"])
    index = int(capsys.readouterr().out.strip())
    hours = [p.hours for p in P.parts()]
    assert sum(hours[:index]) >= 0.2 * sum(hours) > sum(hours[: index - 1])
    arguments = P.argparse.Namespace(family=None, down_to=None, start=3, to=6)
    assert P._indices(arguments) == [3, 4, 5]
    arguments = P.argparse.Namespace(family=None, down_to=len(hours) - 3, start=0, to=None)
    assert P._indices(arguments) == [len(hours) - 1, len(hours) - 2, len(hours) - 3]


def test_a_root_stamped_with_another_key_is_refused(tmp_path: Path) -> None:
    P.claim(tmp_path)
    stamp = tmp_path / "KEY.json"
    assert json.loads(stamp.read_text())["key"] == P.key()
    P.claim(tmp_path)
    stamp.write_text(json.dumps({"key": "another"}))
    with pytest.raises(SystemExit, match="holds parts of key another"):
        P.claim(tmp_path)


def _row(label: str, bias: float, sigma: float, n_kappa: int = 1_000) -> dict[str, object]:
    return {
        "label": label,
        "bias": bias,
        "sigma": sigma,
        "positive_n_kappa": n_kappa,
        "support": 1e-3,
        "sigma_curve": sigma,
    }


def _scan(laws: tuple[tuple[int, Law], ...] = ()) -> P.Scan:
    return P.Scan("S9:family", 5, L_R, 4_000, laws, D.NMAX, "cell", lambda K, law: [])


def test_the_candidate_rule_picks_by_stage() -> None:
    """A: first in order at n'; B: smallest n; C: the first law with a pick; else stop."""
    scan = _scan()
    # d at 4,000 for bias 0.03 and sigma 0.5 is 3.79: stage A takes the second candidate.
    readings = {(5, L_R.key): {"rows": [_row("O1", 0.001, 0.5), _row("O2", 0.03, 0.5)]}}
    assert P.pick(scan, readings)["stage"] == "A"
    assert P.pick(scan, readings)["label"] == "O2"
    # Stage B: (3 * 1 / 0.02)^2 = 22,500 -> 24,000 beats (3 * 1 / 0.015)^2 = 40,000.
    readings = {(5, L_R.key): {"rows": [_row("O1", 0.015, 1.0), _row("O2", 0.02, 1.0)]}}
    assert P.pick(scan, readings) == {
        "stage": "B",
        "label": "O2",
        "n": 24_000,
        "law": L_R.key,
        "K": 5,
        "positive_n": 4_000,
    }
    other = Law(0.5, 0.0, 3.5)
    scan = _scan(((5, other), (4, L_R)))
    readings = {
        (5, L_R.key): {"rows": [_row("O1", 1e-5, 1.0)]},
        (5, other.key): {"rows": [_row("O1", 1e-5, 1.0)]},
        (4, L_R.key): {"rows": [_row("O1", 0.05, 1.0, 9_000)]},
    }
    chosen = P.pick(scan, readings)
    assert (chosen["stage"], chosen["K"], chosen["n"], chosen["positive_n"]) == (
        "D",
        4,
        12_000,
        12_000,
    )
    del readings[(4, L_R.key)]
    assert P.pick(scan, readings)["stage"] == "incomplete"


def test_merge_refuses_a_missing_part_or_a_wrong_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit, match="parts missing"):
        P.merge(tmp_path)
    part = next(p for p in P.parts() if p.family == "mc")
    start, stop = part.detail
    rows = [
        {"stream": part.name, "replicate": r, "seed": D.stream_seed(part.name, r) + (r == 3)}
        for r in range(start, stop)
    ]
    (tmp_path / "parts").mkdir()
    frame = pd.DataFrame(rows).assign(
        estimand="",
        truth=0.0,
        estimate=0.0,
        std_error=0.0,
        ci_lower=0.0,
        ci_upper=0.0,
        covered=1,
        failure="",
    )
    frame.to_csv(tmp_path / "parts" / part.filename, index=False)
    monkeypatch.setattr(P, "parts", lambda: (part,))
    with pytest.raises(SystemExit, match="carry another seed"):
        P._read_sample(tmp_path, "mc")
    frame.loc[3, "seed"] = D.stream_seed(part.name, 3)
    frame.to_csv(tmp_path / "parts" / part.filename, index=False)
    assert len(P._read_sample(tmp_path, "mc")) == stop - start
