"""The part-indexed pilot driver: a fixed part list, resumable parts, a merge that checks seeds."""

from __future__ import annotations

import collections
import itertools
import json
from pathlib import Path

import numpy as np
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


# ----------------------------------------------------------------------------- rev 4.2: stability
#: The digest of the first ``REV_41_PARTS`` idents with their file names, and of the seeds of the
#: first and last replicate of each of their sample parts, at rev 4.1 (``ef4d7a52``).
REV_41_IDENTS = "30896f39c2de7b5e2f494f3ae94197afd3a3f401e69376aa1325e36a4369b5a2"
REV_41_SEEDS = "ea730f5068948dd787f193dce4157282f43fd8dcc7a35312c8c61a0847a86f16"


def test_the_rev_41_parts_keep_their_idents_order_and_seeds() -> None:
    """Rev 4.2 appends parts.  The first 3,288 idents, their order and their seeds are rev 4.1's."""
    import hashlib

    every = P.parts()
    assert P.REV_41_PARTS == 3_288
    first = every[: P.REV_41_PARTS]
    idents = hashlib.sha256()
    for part in first:
        idents.update(f"{part.ident}\n{part.filename}\n".encode())
    assert idents.hexdigest() == REV_41_IDENTS
    seeds = hashlib.sha256()
    for part in first:
        if part.family in P.SAMPLE_FAMILIES:
            start, stop = part.detail
            for replicate in (start, stop - 1):
                seeds.update(
                    f"{part.name}|{replicate}|{D.stream_seed(part.name, replicate)}\n".encode()
                )
    assert seeds.hexdigest() == REV_41_SEEDS
    rev_42 = every[P.REV_41_PARTS : P.REV_42_PARTS]
    assert {part.family for part in rev_42} == {"band", "rung", "witness"}
    # Exact parts of the new family come last, after every sample part.
    appended = [part.family for part in rev_42]
    assert appended.index("witness") > max(i for i, f in enumerate(appended) if f != "witness")


#: The digests of the first ``REV_42_PARTS`` idents with their file names, and of the seeds of the
#: first and last replicate of each of their sample parts, at rev 4.2 (head ``956f962d``).
REV_42_IDENTS = "035433152856f0f163a08a6ecab17ed7a80631dcb9081b6b77f0104d4f5eac79"
REV_42_SEEDS = "71ed2419bf0fc980ac50325185646cce76b8d71790c803defe47cfe3831b34a4"


def test_the_rev_43_parts_append_after_every_rev_42_part() -> None:
    """Rev 4.3 moves none of the first 3,449 parts; the new families follow in the declared order."""
    import hashlib

    every = P.parts()
    assert P.REV_42_PARTS == 3_449
    idents, seeds = hashlib.sha256(), hashlib.sha256()
    for part in every[: P.REV_42_PARTS]:
        idents.update(f"{part.ident}\n{part.filename}\n".encode())
        if part.family in P.SAMPLE_FAMILIES:
            start, stop = part.detail
            for replicate in (start, stop - 1):
                seeds.update(
                    f"{part.name}|{replicate}|{D.stream_seed(part.name, replicate)}\n".encode()
                )
    assert idents.hexdigest() == REV_42_IDENTS
    assert seeds.hexdigest() == REV_42_SEEDS
    new = every[P.REV_42_PARTS :]
    order = [part.family for part in new]
    # Four factor-2 scans, then truth_boot, band_boot and f2_c1, each family contiguous.
    assert order == (
        ["lawscan"] * 4
        + ["truth_boot"] * order.count("truth_boot")
        + ["band_boot"] * order.count("band_boot")
        + ["f2_c1"] * order.count("f2_c1")
    )
    assert [order.count(f) for f in ("lawscan", "truth_boot", "band_boot", "f2_c1")] == [
        4,
        92,
        64,
        1_376,
    ]
    assert len(every) == 4_985
    scans = [p for p in new if p.family == "lawscan"]
    assert all(len(p.detail) == 5 and p.detail[4] == D.WALD_REGIME_FACTOR for p in scans)
    # The S3 truth_boot parts come before S1 and S2.
    boots = [p.name.split("|")[1].split(":")[0] for p in new if p.family == "truth_boot"]
    assert boots == sorted(boots, key=lambda study: ("S3", "S1", "S2").index(study))
    assert len({part.ident for part in every}) == len(every)


def test_the_bootstrap_streams_are_the_declared_pilots() -> None:
    streams = D.bootstrap_streams()
    by_family = collections.Counter(s.family for s in streams)
    assert by_family == {"truth_boot": 23, "band_boot": 2, "f2_c1": 43}
    assert {s.replicates for s in streams if s.family == "truth_boot"} == {400}
    assert {s.replicates for s in streams if s.family != "truth_boot"} == {3_200}
    # Only a single-fold, non-MSM scenario bootstraps.
    for stream in streams:
        if stream.family == "truth_boot":
            study, name = stream.scenario
            scenario = next(s for s in D.scenarios(study) if s.name == name)
            assert scenario.n_folds == 1 and not scenario.msm
    assert {s.name for s in streams if s.family == "band_boot"} == {"S1:band_k5", "S2:band_k5"}
    # The earlier streams are the rev 4.2 ones: f2 keeps the 65 rev 4.2 cells.
    assert sum(s.family == "f2" for s in D.streams()) == 65
    # The unmoved cells' rev 4.2 f2 parts are the other half of the F2 reading.
    assert len(D.unmoved_f2_idents()) == 65 - by_family["f2_c1"]
    assert all(s.cost() > 0 for s in streams)


def test_list_prints_the_cpu_hours_of_each_family(capsys: pytest.CaptureFixture[str]) -> None:
    P.main(["list", "--family", "lawscan"])
    out = capsys.readouterr().out
    assert "lawscan" in out and "CPU-h" in out
    P.main(["list"])
    out = capsys.readouterr().out
    for family in ("truth_boot", "band_boot", "f2_c1"):
        assert any(line.split()[:1] == [family] for line in out.splitlines()), family


def test_the_new_families_are_the_declared_pilots() -> None:
    streams = {s.ident: s for s in D.streams() if s.family in ("band", "rung")}
    band = [s for s in streams.values() if s.family == "band"]
    assert [(s.name, s.n, s.replicates) for s in band] == [
        ("S1:band_k5", 76_000, 3_200),
        ("S2:band_k5", 64_000, 3_200),
    ]
    assert all(len(s.plans) == 6 for s in band)
    rung = [s for s in streams.values() if s.family == "rung"]
    assert len(rung) == 9
    assert {s.n for s in rung} == set(D.ROOT_N_SIZES)
    assert {s.replicates for s in rung} == {1_000}
    witnesses = [p for p in P.parts() if p.family == "witness"]
    assert len(witnesses) == len(set(D.FALLBACKS.values())) == 7


# ----------------------------------------------------------------------------- rev 4.2: equivalence
def _csv(rows: pd.DataFrame) -> bytes:
    return rows.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode()


def _sample_rows(part: P.Part) -> pd.DataFrame:
    start, stop = part.detail
    rng = pd.Series(range(start, stop), dtype=float)
    return pd.DataFrame(
        {
            "stream": part.name,
            "replicate": list(range(start, stop)),
            "seed": [D.stream_seed(part.name, r) for r in range(start, stop)],
            "n": 4_000,
            "estimand": "",
            "truth": 0.25,
            "estimate": 0.25 + 0.01 * ((rng * 7) % 11 - 5) / 5,
            "std_error": 0.01 + 0.0001 * rng,
            "ci_lower": 0.2,
            "ci_upper": 0.3,
            "covered": 1,
            "failure": "",
        }
    )


def _mc_part() -> P.Part:
    return next(p for p in P.parts() if p.family == "mc")


def test_check_rtol_passes_ulp_noise_and_fails_a_change_of_1e_6() -> None:
    part = _mc_part()
    frame = _sample_rows(part)
    stored = _csv(frame)
    noisy = frame.copy()
    noisy.loc[3, "estimate"] = np.nextafter(noisy.loc[3, "estimate"], 1.0)
    noisy.loc[5, "std_error"] = np.nextafter(noisy.loc[5, "std_error"], 1.0)
    assert _csv(noisy) != stored
    found = P.compare_part(part, _csv(noisy), stored, 1e-9)
    assert found.same
    assert 0 < found.max_relative_difference < 1e-15
    moved = frame.copy()
    moved.loc[3, "estimate"] *= 1 + 1e-6
    found = P.compare_part(part, _csv(moved), stored, 1e-9)
    assert not found.same
    assert found.max_relative_difference == pytest.approx(1e-6, rel=1e-3)
    assert "estimate" in found.detail
    # The same change passes at a looser tolerance: the tolerance is the only difference.
    assert P.compare_part(part, _csv(moved), stored, 1e-5).same


def test_check_rtol_reads_integers_text_and_zero_exactly() -> None:
    """A seed one off is 2e-10 of a 32-bit seed: it must not pass as noise.  atol is zero."""
    part = _mc_part()
    frame = _sample_rows(part)
    stored = _csv(frame)
    for column, value in (("seed", frame.loc[2, "seed"] + 1), ("covered", 0), ("stream", "x")):
        changed = frame.copy()
        changed.loc[2, column] = value
        assert not P.compare_part(part, _csv(changed), stored, 1e-9).same, column
    zero = frame.copy()
    zero["truth"] = 0.0
    zeroed = zero.copy()
    zeroed.loc[1, "truth"] = 1e-300
    assert not P.compare_part(part, _csv(zeroed), _csv(zero), 1e-9).same
    assert P.compare_part(part, _csv(zero), _csv(zero), 1e-9).same
    short = frame.iloc[:-1]
    assert not P.compare_part(part, _csv(short), stored, 1e-9).same


def test_check_rtol_compares_a_json_part_value_by_value() -> None:
    part = next(p for p in P.parts() if p.family == "witness")
    stored = {"rows": [{"node": 1, "epsilon": 0.0102, "valid": True, "rung": "D1"}], "law": [1.0]}

    def dump(payload: dict[str, object]) -> bytes:
        return json.dumps(payload, sort_keys=True).encode()

    noisy = {"rows": [{**stored["rows"][0], "epsilon": 0.0102 * (1 + 1e-12)}], "law": [1.0]}  # type: ignore[index]
    assert P.compare_part(part, dump(noisy), dump(stored), 1e-9).same
    moved = {"rows": [{**stored["rows"][0], "epsilon": 0.0102 * (1 + 1e-6)}], "law": [1.0]}  # type: ignore[index]
    assert not P.compare_part(part, dump(moved), dump(stored), 1e-9).same
    for field, value in (("node", 2), ("valid", False), ("rung", "D2")):
        other = {"rows": [{**stored["rows"][0], field: value}], "law": [1.0]}  # type: ignore[index]
        assert not P.compare_part(part, dump(other), dump(stored), 1e-9).same, field


def test_every_kind_has_a_part_and_the_picker_takes_the_cheapest() -> None:
    chosen = P.pick_kinds()
    assert tuple(chosen) == P.KINDS
    for kind, index in chosen.items():
        assert P.kind_of(P.parts()[index]) == kind
        same = [p.hours for p in P.parts() if P.kind_of(p) == kind]
        assert P.parts()[index].hours == min(same), kind
    inside = P.pick_kinds(0, P.REV_41_PARTS)
    assert set(inside) == set(P.KINDS) - {"band", "ltmle_boot", "sdr_boot", "band_boot"}
    assert all(index < P.REV_41_PARTS for index in inside.values())
    assert P.pick_kinds(3_300, 3_310).keys() <= {"band", "ltmle"}


def test_the_kinds_command_prints_the_indices(capsys: pytest.CaptureFixture[str]) -> None:
    P.main(["kinds", "--from", "0", "--to", "20"])
    out = capsys.readouterr().out
    assert "check " in out
    assert "exact_scan" in out
    assert "no part in the range for kinds" in out


# ----------------------------------------------------------------------------- rev 4.2: merge
@pytest.fixture
def small_list(monkeypatch: pytest.MonkeyPatch) -> tuple[P.Part, P.Part]:
    """A part list of one sample part and one exact part; one family is read."""
    sample = _mc_part()
    lawscan = next(p for p in P.parts() if p.family == "lawscan")
    monkeypatch.setattr(P, "parts", lambda: (lawscan, sample))
    monkeypatch.setattr(P, "SAMPLE_FAMILIES", ("mc",))
    monkeypatch.setattr(P, "READERS", {"mc": lambda frame: []})
    return lawscan, sample


def _lawscan_bytes(part: P.Part) -> bytes:
    q = next(q for q in D.FALLBACKS if str(q) == part.name)
    payload = {"quantity": part.name, "pick": list(D.FALLBACKS[q].key), "rows": []}
    return (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()


def _stamp(root: Path, stamped: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "KEY.json").write_text(json.dumps({"key": stamped}))


def _fill(root: Path, parts: tuple[P.Part, ...]) -> None:
    for part in parts:
        data = _csv(_sample_rows(part)) if part.family == "mc" else _lawscan_bytes(part)
        P._write(root / "parts" / part.filename, data)


def _check_into(
    root: Path, monkeypatch: pytest.MonkeyPatch, factor: float, indices: tuple[int, ...]
) -> list[bool]:
    """Run ``check`` on ``indices`` with a recompute that scales every estimate by ``factor``."""
    every = P.parts()

    def compute(part: P.Part, scratch: Path, jobs: int) -> bytes:
        if part.family != "mc":
            return _lawscan_bytes(part)
        frame = _sample_rows(part)
        frame["estimate"] = frame["estimate"] * factor
        return _csv(frame)

    monkeypatch.setattr(P, "compute", compute)
    assert all(0 <= i < len(every) for i in indices)
    return [P.check(root, i, 1, 1e-9) for i in indices]


def test_check_writes_an_equivalence_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_list: tuple[P.Part, P.Part]
) -> None:
    root = tmp_path / "cloud"
    _stamp(root, "oldkey")
    _fill(root, small_list)
    assert _check_into(root, monkeypatch, 1 + 1e-13, (1,)) == [True]
    (path,) = sorted((root / "equivalence").glob("*.json"))
    record = json.loads(path.read_text())
    assert record["part"] == 1
    assert (record["stored_key"], record["current_key"]) == ("oldkey", P.key())
    assert (record["kind"], record["result"], record["rtol"]) == ("plugin", "pass", 1e-9)
    assert 0 < record["max_relative_difference"] < 1e-12
    assert _check_into(root, monkeypatch, 1 + 1e-6, (1,)) == [False]
    failed = json.loads(next((root / "equivalence").glob("*.json")).read_text())
    assert failed["result"] == "fail"
    assert failed["max_relative_difference"] == pytest.approx(1e-6, rel=1e-2)


def test_merge_refuses_a_foreign_root_without_an_accepted_key_or_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_list: tuple[P.Part, P.Part]
) -> None:
    target, cloud = tmp_path / "target", tmp_path / "cloud"
    _stamp(cloud, "oldkey")
    _fill(cloud, small_list)
    with pytest.raises(SystemExit, match="pass --accept-key oldkey"):
        P.merge(target, [cloud])
    with pytest.raises(SystemExit, match="no passing equivalence record"):
        P.merge(target, [cloud], ["oldkey"])
    # A record of a failed check covers nothing.
    _check_into(cloud, monkeypatch, 1 + 1e-6, (1,))
    with pytest.raises(SystemExit, match="no passing equivalence record"):
        P.merge(target, [cloud], ["oldkey"])


def test_merge_accepts_a_foreign_root_whose_records_cover_every_kind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    small_list: tuple[P.Part, P.Part],
    capsys: pytest.CaptureFixture[str],
) -> None:
    target, cloud = tmp_path / "target", tmp_path / "cloud"
    _stamp(cloud, "oldkey")
    _fill(cloud, small_list)
    # One kind covered: the other kind still refuses, and names itself.
    _check_into(cloud, monkeypatch, 1 + 1e-13, (1,))
    with pytest.raises(SystemExit, match="law_scan"):
        P.merge(target, [cloud], ["oldkey"])
    _check_into(cloud, monkeypatch, 1 + 1e-13, (0,))
    assert P.merge(target, [cloud], ["oldkey"]) is True
    assert (target / "merged" / "mc.csv.gz").exists()
    assert P.stamped_key(target) == P.key()
    assert "== mc:" in capsys.readouterr().out
    # The target keeps only its own key: no part moved into it by hand.
    assert not list((target / "parts").glob("*"))


def test_merge_reads_a_part_from_the_current_key_target_before_a_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_list: tuple[P.Part, P.Part]
) -> None:
    target, share = tmp_path / "target", tmp_path / "share"
    lawscan, sample = small_list
    P.claim(target)
    P.claim(share)
    _fill(target, (lawscan,))
    _fill(share, (sample,))
    assert P.merge(target, [share]) is True
    with pytest.raises(SystemExit, match="1 parts missing"):
        P.merge(tmp_path / "empty", [target])


def test_merge_refuses_a_record_for_a_part_that_changed_since(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_list: tuple[P.Part, P.Part]
) -> None:
    target, cloud = tmp_path / "target", tmp_path / "cloud"
    _stamp(cloud, "oldkey")
    _fill(cloud, small_list)
    _check_into(cloud, monkeypatch, 1 + 1e-13, (0, 1))
    assert P.merge(target, [cloud], ["oldkey"]) is True
    # A hand edit of the stored part voids its record.
    sample = small_list[1]
    path = cloud / "parts" / sample.filename
    path.write_bytes(path.read_bytes().replace(b"0.25", b"0.26", 1))
    with pytest.raises(SystemExit, match="no passing equivalence record"):
        P.merge(target, [cloud], ["oldkey"])


def test_merge_reports_a_stopped_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_list: tuple[P.Part, P.Part]
) -> None:
    monkeypatch.setattr(P, "READERS", {"mc": lambda frame: ["S1 band: R none -> stop"]})
    P.claim(tmp_path)
    _fill(tmp_path, small_list)
    assert P.merge(tmp_path) is False


# ----------------------------------------------------------------------------- rev 4.2: readings
def test_the_band_pilots_are_the_property_phases_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rev 4.2 band pilot is the property phase's ``ic_band`` row; ``band_boot`` its band row.

    Both pilots draw one sample.  The sup-t band reads B = 399 resamples; the test cuts that to
    30 so that the check stays in the fast tier.
    """
    import dataclasses
    import warnings

    from tests.studies import long_horizon_properties as properties
    from tests.studies.evidence.simultaneous import joint_cells

    monkeypatch.setattr(D, "B_GATE", 30)
    stream = next(s for s in D.streams() if s.family == "band" and s.name.startswith("S1"))
    boot = next(s for s in D.bootstrap_streams() if s.family == "band_boot" and "S1" in s.name)
    small = dataclasses.replace(stream, n=2_000)
    small_boot = dataclasses.replace(boot, n=2_000)
    drawn = P._draw(small, (0, 11))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = P._sample_fit(small, (0, 11), drawn)
        new = P._sample_fit(small_boot, (0, 11), drawn)
        fit_set = properties.fit_sets("S1")[
            [f.family for f in properties.fit_sets("S1")].index("simultaneous_coverage")
        ]
        rows = properties.fit_key(("S1", fit_set.name, 0, 1, 2_000), drawn)
    band_cell, control_cell = joint_cells("k5")
    by_cell = {row["cell"]: row for row in rows}
    ic = by_cell[properties.ic_band_cell("k5")]
    assert stream.config is not None
    assert (stream.config.law, stream.n, stream.plans) == (
        fit_set.law,
        fit_set.n,
        fit_set.arms[0].plans,
    )
    assert (boot.config, boot.n, boot.plans) == (stream.config, stream.n, stream.plans)
    (row,) = old
    assert row["covered"] == ic["covered"]
    assert row["pointwise_covered"] == by_cell[control_cell]["covered"]
    assert row["estimate"] == pytest.approx(ic["estimate"])
    assert row["std_error"] == pytest.approx(ic["std_error"])
    (row,) = new
    assert row["covered"] == by_cell[band_cell]["covered"]
    assert row["ic_covered"] == ic["covered"]
    assert row["ic_critical"] == pytest.approx(ic["std_error"])
    assert row["std_error"] == pytest.approx(by_cell[band_cell]["std_error"])
    assert row["pointwise_covered"] == by_cell[control_cell]["covered"]
    # The bootstrap critical value is the band's own: it is not the IC band's.
    assert row["std_error"] != pytest.approx(row["ic_critical"])


def _band_frame(study: str, coverage: float, replicates: int = 3_200) -> pd.DataFrame:
    stream = next(s for s in D.streams() if s.family == "band" and s.name.startswith(study))
    covered = np.zeros(replicates, dtype=int)
    covered[: round(coverage * replicates)] = 1
    return pd.DataFrame(
        {
            "stream": stream.ident,
            "replicate": range(replicates),
            "seed": 0,
            "n": stream.n,
            "estimand": "",
            "truth": 0.0,
            "estimate": 1.0,
            "std_error": 2.0,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "covered": covered,
            "pointwise_covered": covered,
            "failure": "",
        }
    )


def test_the_band_reading_sizes_r_under_the_two_sided_rule_and_stops_without_a_rung() -> None:
    ladder = D.BAND_LADDER
    # Joint coverage 0.95 passes with probability above 0.99 at the smallest rung.
    reading = P.read_band(_band_frame("S1", 0.95))
    assert any(line.startswith("S1 band: R 2000") for line in reading), reading
    # 0.94 needs a larger R; the two-sided rule, not the one-sided floor, sets it.
    assert D.band_pass(0.94, 2_000) < D.BAND_TARGET <= D.band_pass(0.94, 4_000)
    assert any(line.startswith("S1 band: R 4000") for line in P.read_band(_band_frame("S1", 0.94)))
    # 0.91 never reaches the target: the reading stops.
    assert max(D.band_pass(0.91, R) for R in ladder) < D.BAND_TARGET
    stopped = P.read_band(_band_frame("S2", 0.91))
    assert any("none -> stop" in line for line in stopped)


def test_the_band_pass_probability_is_the_two_sided_rule() -> None:
    """Not the one-sided floor: 0.9293 passes the floor rule at R = 2,000, and fails this one."""
    assert D.band_pass(0.9293, 2_000) < 0.5
    # At 0.95 the exact interval is inside (0.92, 0.98) at R = 2,000 with high probability;
    # at 0.97 the upper limit leaves the band.
    assert D.band_pass(0.95, 2_000) > 0.99
    assert D.band_pass(0.975, 2_000) < 0.5
    # It agrees with the shipped interval count by count.
    from tests.studies.evidence.inference import clopper_pearson

    low, high = D._clopper_pearson(np.arange(0, 41), 40, 0.99)
    for k in range(41):
        shipped = clopper_pearson(k, 40, confidence_level=0.99)
        assert (low[k], high[k]) == pytest.approx((shipped.low, shipped.high))


def _rung_frame(study: str, scale: float) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    frames = []
    for stream in (s for s in D.streams() if s.family == "rung" and s.name.startswith(study)):
        sd = scale / np.sqrt(stream.n)
        estimate = 0.3 + rng.normal(0, sd, 1_000)
        frames.append(
            pd.DataFrame(
                {
                    "stream": stream.ident,
                    "replicate": range(1_000),
                    "seed": 0,
                    "n": stream.n,
                    "estimand": "",
                    "truth": 0.3,
                    "estimate": estimate,
                    "std_error": sd,
                    "ci_lower": estimate - 1.96 * sd,
                    "ci_upper": estimate + 1.96 * sd,
                    "covered": (np.abs(estimate - 0.3) <= 1.96 * sd).astype(int),
                    "failure": "",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_the_rung_reading_names_the_control_rung_and_the_slope_rows() -> None:
    lines = P.read_rung(_rung_frame("S1", 1.2))
    control = next(line for line in lines if "n_4000" in line)
    assert "control rung, no claim at R = 1,000" in control
    positive = [line for line in lines if "n_12000" in line or "n_48000" in line]
    assert len(positive) == 2
    assert all("coverage clause pass probability" in line for line in positive)
    slopes = [line for line in lines if "root_n_rate" in line]
    assert len(slopes) == 2
    assert all("pass probability 1.000" in line for line in slopes), slopes


def test_the_slope_pass_probability_fails_a_flat_ladder() -> None:
    rng = np.random.default_rng(5)
    rungs = {
        n: pd.DataFrame(
            {
                "estimate": 0.3 + rng.normal(0, 0.01, 1_000),
                "std_error": 0.01,
                "truth": 0.3,
            }
        )
        for n in D.ROOT_N_SIZES
    }
    assert D.slope_pass(rungs, "spread", 1_000, draws=10, bootstrap=500) == 0.0


def _truth_frame(
    estimands: dict[str, float], family: str = "truth", name: str = "S3:sdr_policy_k5"
) -> pd.DataFrame:
    """Truth-row pilots of one scenario: each estimand with its coverage."""
    streams = D.streams() if family == "truth" else D.bootstrap_streams()
    stream = next(s for s in streams if s.family == family and s.name == name)
    rng = np.random.default_rng(2)
    frames = []
    for estimand, coverage in estimands.items():
        covered = (rng.random(400) < coverage).astype(int)
        frame = pd.DataFrame(
            {
                "stream": stream.ident,
                "replicate": range(400),
                "seed": 0,
                "n": stream.n,
                "estimand": estimand,
                "truth": 0.3,
                "estimate": 0.3 + rng.normal(0, 0.01, 400),
                "std_error": 0.01,
                "ci_lower": 0.0,
                "ci_upper": 1.0,
                "covered": covered,
                "failure": "",
            }
        )
        if family == "truth_boot":
            frame["interval"] = "bca" if estimand.endswith("__bca") else "wald"
            frame["n_failed"] = (np.arange(400) % 40 == 0).astype(int)
            frame["sparse_warning"] = (np.arange(400) % 10 == 0).astype(int)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_the_truth_reading_uses_only_the_gated_rows() -> None:
    """Review major 2: a row below its own tail regime reports and must not set R.

    ``ey_regimen[taper]`` runs at 44,000 in ``sdr_policy_k5`` against an n_kappa of 246,603.  It
    under-covers by design.  With it counted, the truth rows stop; without it they reach 1,200.
    A Wald row of the scenario is gated only at twice its n_kappa, so ``taper vs low`` (42,420)
    reports its Wald row and gates its BCa row.
    """
    assert ("S3", "sdr_policy_k5", "ey_regimen[taper]") not in D.GATED_ROWS
    assert ("S3", "sdr_policy_k5", "ate_regimen[taper vs low]") not in D.GATED_ROWS
    assert ("S3", "sdr_policy_k5", "ate_regimen[taper vs low]__bca") in D.GATED_ROWS
    gated_wald = next(r for r in D.GATED_ROWS if r[0] == "S3" and not r[2].endswith("__bca"))
    name = f"S3:{gated_wald[1]}"
    lines = P.read_truth(
        _truth_frame({gated_wald[2]: 0.95, "ey_regimen[unlisted]": 0.60}, name=name)
    )
    assert "S3: 1 gated truth rows, R 1200" in lines
    assert "1 non-gating (stream, estimand) pairs report and set no R" in lines
    # A gated row at the same low coverage does stop the study.
    stopped = P.read_truth(_truth_frame({gated_wald[2]: 0.60}, name=name))
    assert "S3: 1 gated truth rows, R none -> stop" in stopped


def test_the_bootstrapped_truth_reading_gates_the_bca_row_and_counts_failures() -> None:
    bca = "ate_regimen[taper vs low]__bca"
    frame = _truth_frame(
        {bca: 0.95, "ate_regimen[taper vs low]": 0.60, "ey_regimen[taper]__bca": 0.60},
        family="truth_boot",
    )
    lines = P.read_truth_boot(frame)
    assert "S3: 1 gated bootstrapped truth rows, R 1200" in lines
    assert any("dropped refits per replicate mean 0.03" in line for line in lines)
    assert any("SparseInfluenceWarning fires on 0.100" in line for line in lines)
    # The Wald row below the regime and the ungated mean do not set R: a gated BCa row at low
    # coverage is the only thing that stops the study.
    stopped = P.read_truth_boot(_truth_frame({bca: 0.60}, family="truth_boot"))
    assert "S3: 1 gated bootstrapped truth rows, R none -> stop" in stopped


def test_run_refuses_s1_and_s2_truth_boot_until_the_s3_reading_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    every = P.parts()
    s3 = [p for p in every if p.family == "truth_boot" and p.name.split("|")[1].startswith("S3:")]
    other = next(p for p in every if p.family == "truth_boot" and p not in s3)
    assert every.index(s3[-1]) < every.index(other)
    P.s3_gate(tmp_path, s3[0])
    with pytest.raises(SystemExit, match="S3 truth_boot parts are missing"):
        P.s3_gate(tmp_path, other)
    # With every S3 part present, a stopped S3 reading ends the run and a passing one lets it go.
    (tmp_path / "parts").mkdir()
    for part in s3:
        (tmp_path / "parts" / part.filename).write_text("")
    monkeypatch.setattr(P, "_read_sample", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(P, "read_truth_boot", lambda frame: ["S3: R none -> stop"])
    with pytest.raises(SystemExit, match="S3 truth_boot reading stopped"):
        P.s3_gate(tmp_path, other)
    monkeypatch.setattr(P, "read_truth_boot", lambda frame: ["S3: R 1200"])
    P.s3_gate(tmp_path, other)


def test_the_f2_reading_prints_each_arms_coverage_and_its_error() -> None:
    cell = next(c for c in D.calibration_cells() if c.study == "S1")
    stream = next(s for s in D.streams() if s.family == "f2" and s.name == f"S1:{cell.name}")
    rng = np.random.default_rng(4)
    estimate = rng.normal(0.3, 0.01, 800)
    frame = pd.DataFrame(
        {
            "stream": stream.ident,
            "replicate": range(800),
            "seed": 0,
            "n": stream.n,
            "estimand": "",
            "truth": 0.3,
            "estimate": estimate,
            "std_error": 0.0101,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "covered": (rng.random(800) < 0.94).astype(int),
            "failure": "",
        }
    )
    lines = P.read_f2(frame)
    assert "binomial error" in lines[0]
    assert lines[-1].startswith("S1: k 1 F2 R")


def _f2_frame(stream: D.Stream, sparse: int, replicates: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "stream": stream.ident,
            "replicate": range(replicates),
            "seed": 0,
            "n": stream.n,
            "estimand": "",
            "truth": 0.3,
            "estimate": rng.normal(0.3, 0.01, replicates),
            "std_error": 0.0101,
            "ci_lower": 0.0,
            "ci_upper": 1.0,
            "covered": (rng.random(replicates) < 0.94).astype(int),
            "failure": "",
            "sparse_warning": sparse,
        }
    )


def test_the_f2_c1_reading_reports_the_sparse_flag_and_joins_the_unmoved_cells() -> None:
    moved = next(s for s in D.bootstrap_streams() if s.family == "f2_c1" and "S1:" in s.name)
    unmoved_ident = next(i for i in D.unmoved_f2_idents() if "|S1:" in i)
    unmoved = next(s for s in D.streams() if s.ident == unmoved_ident)
    c1 = _f2_frame(moved, sparse=1)
    lines = P.read_f2_c1(c1)
    assert any("SparseInfluenceWarning fires on 1.000" in line for line in lines)
    # A cell whose fit does not carry the flag records -1 and prints no sparse line.
    assert not any(
        "SparseInfluenceWarning" in line for line in P.read_f2_c1(c1.assign(sparse_warning=-1))
    )
    combined = P.read_f2_combined(_f2_frame(unmoved, sparse=-1), c1)
    assert combined[-1].startswith("S1: k 2 F2 R")
    # The moved cell's rev 4.2 f2 stream is not in the joint reading.
    old = next(s for s in D.streams() if s.family == "f2" and s.name == moved.name)
    assert P.read_f2_combined(_f2_frame(old, sparse=-1), c1)[-1].startswith("S1: k 1 F2 R")


def test_a_per_node_witness_part_reports_the_floors_and_the_fallback() -> None:
    """The witness of a fallback law: every node valid, with its rung and its ratio to the floor."""
    from tests.studies import long_horizon_mutations as M

    rows = M.per_node_witness(3, 3, "low", L_R)
    assert [r["node"] for r in rows] == [1, 2, 3]
    assert all(r["valid"] for r in rows)
    assert rows[0]["rung"] == "D3"
    assert all(r["truth_error"] < 1e-8 and r["base_score"] < 1e-9 for r in rows)
    assert all(r["floors_met"] for r in rows)
    # The mutation is the one the fast tier runs: an update skipped at node t moves the estimate.
    assert all(r["move"] > 5e-4 and r["skipped_score"] > 5e-6 for r in rows)


def test_the_witness_part_of_a_fallback_law_is_valid_at_every_node() -> None:
    """Plan section 4.2 runs the per-node witness on every fallback law, not on ``L_R`` alone."""
    part = next(p for p in P.parts() if p.family == "witness" and "(1, 0, 4.5)" in p.name)
    payload = json.loads(P.compute(part, Path("."), 1))
    rows = payload["rows"]
    assert payload["law"] == [1.0, 0.0, 4.5, 1.0]
    assert {(r["K"], r["m"], r["plan"]) for r in rows} == {
        (4, 3, plan) for plan in ("low", "high", "respond", "mix")
    }
    assert len(rows) == 16 and all(r["valid"] for r in rows)
    assert "16 (plan, node) witnesses" in P.read_witness_text(payload)
