# Incremental-intervention evidence artifacts

This directory contains the registered comparison with R `npcausal` at commit `56a5ac1`, the
reference implementation published with Kennedy (2019). Its influence values carry the
derivative through the treatment mechanism, so the comparison gates inference as well as the
point curve.

Two upstream limits shape the runner, and neither is patched:

- `nsplits = 1` is documented and not implemented, so the reference cross-fits over two folds
  while `cleverly` does not.
- `return_ifvals = TRUE` subtracts a length-`k` vector from an `n`-by-`k` matrix, which R
  recycles down columns. The runner calls `ipsi` once per multiplier, where `k` is one and the
  subtraction is correct.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.npcausal_incremental.regenerate --replicates 4 --n 200 --skip-properties --output build/incremental-smoke --allow-failures
```

Run `python -m tests.canonical.npcausal_incremental.regenerate` to rebuild the committed artifacts.
