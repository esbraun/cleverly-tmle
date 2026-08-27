# Known-stochastic-regime evidence artifacts

This directory contains the registered comparison with pinned R `lmtp` 1.5.4. The runner fits the
static regime and the declared stochastic tilt on each realized Python sample.

`lmtp`'s public interface takes a shift function of the natural treatment value. A known
stochastic regime ignores that value, which is why this study once reported no comparator. The
shared point adapter does not use that interface: it supplies the density ratio and the shifted
frame directly. The static regime uses `mtp = FALSE`. The tilt draws one treatment value per unit
from the declared density and uses `mtp = TRUE`.

Regenerate a disposable smoke study before the declared run:

```console
python -m tests.canonical.stochastic_regimes.regenerate --replicates 4 --n 200 --skip-properties --output build/stochastic-smoke --allow-failures
```

Run `python -m tests.canonical.stochastic_regimes.regenerate` to rebuild the committed artifacts.
