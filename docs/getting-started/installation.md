# Installation

## Supported Python

`cleverly` requires Python 3.11 or newer. It is not published on PyPI yet, so install the current
package from GitHub or from a checked-out commit.

```bash
python -m pip install "git+https://github.com/esbraun/cleverly-tmle.git"
```

The core install includes NumPy, SciPy, scikit-learn, narwhals, joblib, and threadpoolctl. Install
the `all` extra for pandas, polars, LightGBM, and plotting support:

```bash
python -m pip install "cleverly[all] @ git+https://github.com/esbraun/cleverly-tmle.git"
```

For a reproducible analysis, replace the default branch with a release tag or full commit hash.
Because the project is alpha software, an unconstrained Git install can receive public-API changes.

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
