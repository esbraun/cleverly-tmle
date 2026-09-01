# Installation

## Supported Python

`cleverly` requires Python 3.11 or newer. Install the core package from PyPI:

```bash
python -m pip install cleverly
```

The core install includes NumPy, SciPy, scikit-learn, narwhals, joblib, and threadpoolctl. Install
the `all` extra for pandas, polars, and plotting support. Install third-party nuisance estimators
such as XGBoost or LightGBM separately and pass their sklearn-compatible objects directly:

```bash
python -m pip install "cleverly[all]"
```

Pin the complete version for a reproducible analysis:

```bash
python -m pip install "cleverly[all]==0.1.0"
```

`cleverly` uses `0.1.N` versions while it remains alpha software. A later patch can contain a
public API change.

## Development snapshot

Install a reviewed Git snapshot when you need a change that has no release:

```bash
python -m pip install \
  "cleverly @ git+https://github.com/esbraun/cleverly-tmle.git@FULL_COMMIT_HASH"
```

Use a full commit hash. A branch name can move after you record an analysis.

## Development install

Clone the repository and create an editable environment with `uv`:

```bash
git clone https://github.com/esbraun/cleverly-tmle.git
cd cleverly-tmle
uv venv
uv pip install -e ".[dev]"
```

Build this documentation site with:

```bash
nox -s docs
```

The build writes generated HTML to `docs/_build/html/`; that directory is not source and is not
committed.

## Verify the import

```python
import cleverly

print(cleverly.__version__)
```

If an editable install points at another worktree, the test suite will refuse to run and name both
paths. Reinstall the current checkout with `uv pip install -e ".[dev]"`.
