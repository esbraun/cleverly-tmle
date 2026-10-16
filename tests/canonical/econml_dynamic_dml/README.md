# EconML DynamicDML parity fixture

This directory holds the comparator for `StructuralNestedMean`. The comparator is EconML 0.17.0
`DynamicDML`. The library and the test suite never install or import EconML.

| file | role |
| --- | --- |
| `requirements.txt` | Hash-pinned environment. The `econml` entry holds one hash, the cp313 manylinux2014 x86_64 wheel. |
| `Dockerfile` | Builds that environment on a pinned `python:3.13-slim-bookworm` digest. |
| `generate_parity_fixture.py` | Draws the parity cases, fits EconML, and writes the fixture. |
| `run_generator.sh` | Builds the container and runs the generator in it. |
| `parity_fixture.json` | The committed inputs, folds, and EconML coefficients, covariances and contrast errors. |

## Regenerating the fixture

Run `./run_generator.sh`. The generator records its own sha256, the Dockerfile sha256, the
requirements sha256, and the wheel sha256 in the fixture. It also records the Python version and
the route under `environment`. The test `tests/unit/test_structural_econml_parity.py` fails when
the committed generator differs from the recorded one. Regenerate the fixture after any edit to
the generator.

The committed fixture did not come from the container. Its `environment` entry names a virtual
environment whose installed distributions equal the pins of `requirements.txt`, on Python 3.13.16.
A second run in that environment reproduces the committed numbers byte for byte. The container
has not been built. Build it once and regenerate the fixture before the comparator phase, so that
the recorded route is `container`.

The fixture holds no study replicates. The full comparator study runs later and has its own
manifest.

## What the cases show

Each case fits both sides on the same data, the same folds, and the same nuisance models. The
test compares them.

| quantity | result |
| --- | --- |
| coefficients, every node type | equal to rounding error |
| coefficient errors, scalar nodes | equal after the factor `n / (n - 1)` that `cleverly` uses for the variance |
| coefficient errors, categorical nodes | differ, because EconML fills the off-diagonal Jacobian blocks of a vector coding in the transposed order |
| contrast error, scalar nodes and no features | equal after the same factor |
| contrast error, categorical nodes | the `cleverly` value matches the correct sandwich and a fixed-residual bootstrap |

The generator writes both sandwiches and the bootstrap error for each contrast so that the
orientation claim rests on numbers in the fixture.
