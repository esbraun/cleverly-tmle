# Sequentially doubly robust longitudinal estimation fixture

This registered study fits `SDR` on five paired scenarios and compares each with pinned R `lmtp`
1.5.4 (`cf_sdr` and `theta_dr(is_sdr = TRUE)`). Both sides read the same shifted values, per-node
ratios and fold assignment, written beside each replicate. `tests/studies/canonical_longitudinal_sdr.py`
states the scenarios and the pairing reading. `tests/studies/longitudinal_sdr_properties.py`
states the property families.

Run the pre-declaration pairing gates (G1 and G2 of `X4-plan.md` 6.2) first. They need Docker and
the `cleverly-lmtp-crossfit:1.5.4` image, and they write `gates.csv` and `timing.csv` here:

```bash
python -m tests.canonical.longitudinal_sdr.gates --replicates 3 --n 2000
```

Then run a disposable smoke study:

```bash
python tests/canonical/longitudinal_sdr/regenerate.py --replicates 2 --n 500 --skip-properties --output .tmp/longitudinal-sdr-smoke
```

Regenerate the declared study and refresh its document:

```bash
python tests/canonical/longitudinal_sdr/regenerate.py
python -m tests.studies.evidence.document --slug longitudinal-sdr
```
