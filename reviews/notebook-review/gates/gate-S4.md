# Gate S4: commit d546049f, scientific-notation decimals in the narration check

Verdict: PASS. No required fixes. One non-blocking nit.

## Checks run

| check | result |
| --- | --- |
| `pytest -q -p no:cacheprovider tests/unit/test_documentation_runtime.py -k "narrat or scientific"` (no `-n`) | 12 passed, 0.8 s (includes the new unit test and the per-notebook narrated-decimal test) |
| regex probe over 16 edge strings and 15 tolerance cases against `narrated_decimals`, `narration_mismatches`, `_resolution` | all as specified below |
| TWINS notebook cell 18 stored output vs the three new `UNPRINTED_DECIMALS` reasons | `0.0005`, `0.0000`, `0.0042` printed exactly as the reasons say |
| ledger TW-04 / sweep P10 against the change | the harness half of the TW-04 repair is delivered; the notebook half is N10 (`plan.md:88`, `progress.md:39` pending) |

## CONFIRMED-OK

1. Regex. `6e702`, `1e-10`, `1.5e3abc`, `1.5e` (dangling marker), `1.2e3.4` are all not read. `2.5E+03`, `+1.50e-02`, `-4.2e-3`, Unicode-minus exponent `4.2e−3` (normalised to ASCII), a literal before `,`, `.` or `)`, and one inside a Markdown table cell are read. Table and `$...$` LaTeX literals were already read in fixed-point form before this commit; no regression, and TWINS has no such case. Inline code and URLs stay excluded (the test's `tol=1.5e-08` / `2.1e-3` line proves it).
2. Tolerance and precision. `_resolution` gives 1e-7 for both `5.114e-04` and `0.0005114`, 1e-4 for `4.2e-03`, 100 for `2.5E+03`, 1 for an integer, 1e-10 for `1e-10`. Scientific path: `4.20e-03` vs stored `0.00420` accepted, vs `0.0042` refused (trailing zero is a precision claim); `1.0e3` vs `1049` accepted and vs `1051` refused (half-unit scaled by exponent); a stored `0.00051145` at exactly half a unit is accepted via the `1e-9` slack; higher-precision output (`0.00420000001`) matches a coarser literal. Sign is honoured (`5.114e-04` vs `-5.114e-04` refused).
3. Fixed-point path unchanged. Same `0.5 * 10^-places + 1e-12` absolute tolerance, no output-precision condition; a fixed `0.26` still matches a stored `2.6e-01`. The pre-existing test `test_the_narration_check_reads_precision_and_refuses_a_moved_number` passes unmodified.
4. Failing-first test is meaningful. On the parent commit `narrated_decimals("... 5.114e-04 and 1.23E-04.")` returns `[]`, so the first assertion fails there. Each component has a nonzero witness: tolerance (`4.228e-04` refused against `0.0004227`, `4.227e-04` accepted), precision condition (`4.20e-03` refused against `4.2e-03`), both notations (`0.0005114`), uppercase `E`, Unicode minus, sign, exclusions, and the `unprinted` mapping.
5. TWINS `UNPRINTED_DECIMALS`. Keys are the exact written forms (verified in the notebook source). Reasons are accurate against the stored output. They cannot mask another problem: the mapping is keyed by exact literal, and `test_every_narrated_decimal_matches_a_stored_output` fails on a stale key (`stale = set(unprinted) - set(narrated_decimals(markdown))`), so N10 is forced to delete the three rows when it rewrites the reading. "removed in N10" matches `plan.md` row N10 (explicit number formats, TW-04).
6. Docstrings. Module docstring, `_DECIMAL` comment, `narrated_decimals` and `narration_mismatches` docstrings describe the implemented behaviour, including the no-decimal-point exclusion and the inline-code exclusion. Commit message gate list is consistent with what runs.

## REQUIRED FIXES

None.

## Notes (non-blocking)

- `_resolution` comment "where `10.0 ** n` would raise on stray text" is imprecise: `10.0 ** n` raises `OverflowError` on a large `n`, not on text. The `float("1e...")` form is correct; only the comment wording is off.
