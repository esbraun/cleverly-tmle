# Pinned `lmtp` cross-fit adapter

This directory tests the thin adapter in `tests/canonical/lmtp_crossfit_adapter.R`.
The adapter passes an exact zero-based fold assignment to pinned `lmtp` 1.5.4 internals.

Run the smoke check from the repository root.

```console
docker build -t cleverly-lmtp-crossfit:1.5.4 tests/canonical/lmtp_crossfit
docker run --rm -v "$PWD/tests/canonical:/fixture:ro" cleverly-lmtp-crossfit:1.5.4 /fixture/lmtp_crossfit/smoke.R
```

The adapter does not create a registered evidence row. The ordinary and survival studies must
still generate their complete artifacts before registration.
