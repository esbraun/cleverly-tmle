# Installation

## Supported Python

`cleverly` requires Python 3.11 or newer. Install the core package from PyPI:

```bash
python -m pip install cleverly
```

The core install includes NumPy, SciPy, scikit-learn, narwhals, joblib, cloudpickle, and
threadpoolctl. It holds no data-frame library, so the quickstart and the examples need an extra:

| extra | adds |
| --- | --- |
| `pandas` | pandas tables |
| `polars` | polars tables |
| `plot` | matplotlib, for `cleverly.plotting` |
| `all` | the three above |

```bash
python -m pip install "cleverly[all]"
```

Without a data-frame library, the dataset generators raise an `ImportError` that names
`cleverly[pandas]` and `cleverly[polars]`. Arrow input, a `pyarrow.Table` or an Arrow-backed pandas
frame, needs `pyarrow`, which no extra installs: run `python -m pip install pyarrow` as well.
Install third-party learners such as XGBoost or LightGBM separately and pass their
scikit-learn-compatible objects directly.

Pin the complete version you analysed with, for example 0.2.8:

```bash
python -m pip install "cleverly[all]==0.2.8"
```

A later release can still contain a public API change. A result that one release saves loads in another release with a warning, and
the load does not migrate it.

## Development snapshot

Install a reviewed Git snapshot when you need a change that has no release:

```bash
python -m pip install \
  "cleverly @ git+https://github.com/esbraun/cleverly-tmle.git@FULL_COMMIT_HASH"
```

Use a full commit hash. A branch name can move after you record an analysis.

## Work on cleverly itself

To change `cleverly`, run its tests or build this site, follow
[set up your environment](../development/contributing.md#set-up-your-environment) in the
contributor guide.

## Verify the import

```python
import cleverly

print(cleverly.__version__)
```
