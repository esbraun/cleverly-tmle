# Selector-based multi-arm C-TMLE

This registered study covers Cleverly's greedy, ordered, and discrete joint multi-arm
selector targets. Its equivalence artifact is intentionally empty: pinned R `ctmle`
0.1.2 documents and implements binary treatment only, so separate binary fits would be a
different estimator rather than a comparator.

The row uses the reporting policy. Its property record runs all three selector paths
against one forced empty path on identical draws. A path that stops before adjusting for a
multi-arm confounder reaches an error ratio of one, and that result is published as a
limitation rather than hidden by a surrogate comparison.

Regenerate from the repository root:

```powershell
uv run --extra dev python -m tests.canonical.multi_arm_ctmle_selector.regenerate
```

The reader-facing results are in
[`selector-based-multi-arm-c-tmle.md`](../../../docs/technical-reference/method-evidence/selector-based-multi-arm-c-tmle.md).
