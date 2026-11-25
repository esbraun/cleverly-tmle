"""The stored-text comparison passes one rounding and refuses every moved result.

Without the refusals, a comparator that accepted everything would pass every notebook; without
the acceptances, a library release that rounds one printed digit differently would turn the fast
tier red with nothing to fix.
"""

from __future__ import annotations

from tests.notebook_freshness import stored_cell_text, text_moves, unreproduced

TABLE = "ate      0.1524    0.0209   [0.1115, 0.1927]   <0.0001\nn = 1500\n"


def test_identical_and_rechunked_text_reproduces() -> None:
    assert text_moves(TABLE, TABLE) == []
    assert text_moves(TABLE, TABLE.replace("   ", "  ")) == [], "whitespace runs collapse"


def test_one_last_place_rounding_change_passes() -> None:
    assert text_moves(TABLE, TABLE.replace("0.1927", "0.1928")) == []
    assert text_moves("psi 0.088", "psi 0.0885") == [], "the coarser place sets the unit"


def test_a_moved_number_fails() -> None:
    assert text_moves(TABLE, TABLE.replace("0.1927", "0.1937")) == ["0.1927 -> 0.1937"]
    # PL-S2's shape: one end moved by one unit, the other by two.
    assert text_moves("0.054 to 0.098", "0.053 to 0.100") == ["0.098 -> 0.100"]


def test_several_one_unit_changes_in_one_cell_fail() -> None:
    """PT-S1's shape: three limits each moved by one unit from one rule change."""
    stored = "limits [0.1559, 0.177] [0.0755, 0.252] [0.0308, 0.295]"
    fresh = "limits [0.1558, 0.177] [0.0754, 0.252] [0.0307, 0.295]"
    moves = text_moves(stored, fresh)
    assert len(moves) == 2 and all("second last-place change" in move for move in moves)


def test_changed_words_and_lines_fail() -> None:
    assert text_moves(TABLE, TABLE.replace("ate", "att"))
    assert text_moves(TABLE, TABLE + "extra line\n")
    assert text_moves(TABLE, TABLE.split("\n", 1)[0])


def test_a_count_must_be_equal() -> None:
    """An integer is a count, not a rounded value, so it gets no rounding allowance."""
    assert text_moves("followed 1448", "followed 1449") == ["1448 -> 1449"]


def test_a_digest_or_identifier_compares_as_text() -> None:
    """A hex id is not a run of numbers that each get a rounding allowance."""
    assert text_moves("study 4f60ba6043a32341", "study 4f60ba6043a32342")
    assert text_moves("C=1000000.0,max_iter=2000", "C=1000000.0,max_iter=2001")
    assert text_moves("version 1.5.4", "version 1.5.5")


def test_the_version_line_is_masked_and_nothing_else() -> None:
    assert text_moves("cleverly 0.2.4\n", "cleverly 0.2.8\n") == []
    assert text_moves("cleverly 0.2.4 rows 3\n", "cleverly 0.2.8 rows 3\n")


def test_stored_cell_text_reads_stdout_and_plain_text_and_skips_figures() -> None:
    outputs = [
        {"output_type": "stream", "name": "stdout", "text": "psi "},
        {"output_type": "stream", "name": "stdout", "text": ["0.15", "\n"]},
        {"output_type": "stream", "name": "stderr", "text": "/tmp/ipykernel_1/2.py: Warning\n"},
        {
            "output_type": "display_data",
            "data": {"image/png": "iVBOR", "text/plain": "<Figure size 700x450 with 1 Axes>"},
        },
        {"output_type": "display_data", "data": {"text/plain": "<Figure size 700x450>"}},
        {"output_type": "execute_result", "data": {"text/plain": "frame", "text/html": "<t/>"}},
    ]
    assert stored_cell_text(outputs) == "psi 0.15\n<Figure size 700x450>\nframe\n"


def test_unreproduced_reports_moves_and_stale_exemptions() -> None:
    stored = {"a": "x 0.10", "b": "y 0.20", "c": "z 0.30"}
    fresh = {"a": "x 0.10", "b": "y 0.25", "c": "z 0.30"}
    moved, stale = unreproduced(stored, fresh, {})
    assert moved == {"b": ["0.20 -> 0.25"]} and stale == []
    moved, stale = unreproduced(stored, fresh, {"b": "kernel-only", "c": "no longer needed"})
    assert moved == {} and stale == ["c"]
    _, stale = unreproduced(stored, fresh, {"gone": "cell removed"})
    assert stale == ["gone"]


def test_machine_noise_in_a_fitted_cell_passes_and_a_move_still_fails() -> None:
    # The drift CI showed in sequential-pricing against a stamp from another machine: under the
    # last-place rule alone this is four changes in one cell.
    stored = "kish 5325.470 2922.186 21.835 1973.710"
    assert text_moves(stored, "kish 5325.445 2922.193 21.834 1973.717") == []
    assert text_moves(stored, "kish 5325.470 2922.186 21.835 1993.710") == [
        "1973.710 -> 1993.710"
    ], "a 1 percent move fails"
    assert text_moves("psi 0.1025 0.1154", "psi 0.1026 0.1155") == [
        "0.1154 -> 0.1155 (a second last-place change in one cell)"
    ], "changes a reading would show still count against the one-rounding budget"


def test_a_hardware_sensitive_cell_passes_cpu_drift_and_a_move_still_fails() -> None:
    # The drift a CI runner printed for dose-response's boosted fit, from its stored run: several
    # last-place changes and a 0.2% change in one cell.
    stored = "support 548.1 517.5 360.1 46.8 estimate 0.542 -0.059 1.268"
    fresh = "support 547.9 517.3 359.4 46.9 estimate 0.541 -0.060 1.270"
    assert text_moves(stored, fresh) != [], "the default rule refuses it"
    assert text_moves(stored, fresh, hardware_sensitive=True) == []
    moved = "support 548.1 517.5 366.1 46.8 estimate 0.542 -0.059 1.268"
    assert text_moves(stored, moved, hardware_sensitive=True) == ["360.1 -> 366.1"], (
        "a move of more than a percent still fails"
    )
    assert text_moves("followers 1623", "followers 1622", hardware_sensitive=True) == [
        "1623 -> 1622"
    ], "a count must still be equal"


def test_a_hardware_sensitive_cell_is_stale_only_once_it_is_gone() -> None:
    stored = {"estimate": "psi 0.542", "support": "n 548.1"}
    fresh = {"estimate": "psi 0.541", "support": "n 548.1"}
    hardware = {"estimate": "boosting", "support": "boosting", "gone": "boosting"}
    moved, stale = unreproduced(stored, fresh, {}, hardware)
    assert moved == {} and stale == ["gone"]
