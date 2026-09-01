# Releases

This page defines the release process for `cleverly`. The
[Python Packaging User Guide](https://packaging.python.org/en/latest/) supplies the packaging
standards behind this process.

## Version policy

`src/cleverly/_version.py` is the sole version source. Hatch reads that file when it creates
distribution metadata. The installed package exposes the same value as `cleverly.__version__`.

The first published version is `0.1.0`. Each later alpha release increments only the patch
component to produce `0.1.N`. A release-bearing pull request changes the version before review.

A release-bearing pull request is a change that the maintainer plans to tag after merge. Ordinary
pull requests do not change the version. The maintainer chooses the next policy after the alpha
series ends.

## Review a release

Prepare the release in a pull request from `main`.

1. Change `__version__` in `src/cleverly/_version.py` to the next unused `0.1.N` value.
2. Run every pull request check, including the package checks.
3. Merge the reviewed pull request into `main`.
4. Create the matching `v0.1.N` tag on the reviewed `main` commit.
5. Push the tag, and review both publishing environments in GitHub Actions.

The workflow rejects a tag with another form. It also rejects a tag that differs from the source
version or points outside `main` history.

## Package gates

The package job builds one wheel and one source distribution. It applies these gates before an
artifact can reach a package index.

| gate | evidence |
| --- | --- |
| metadata | `python -m twine check --strict dist/*` |
| archive contents | `python scripts/check_distribution.py dist` |
| wheel install | a clean pandas environment runs `scripts/smoke_backend.py` |
| source install | a clean polars environment runs `scripts/smoke_backend.py` |

The archive check requires the MIT license and the `py.typed` marker. It also excludes the test
tree because `tests/canonical/` has a different license.

## Publish immutable artifacts

The `publish.yml` workflow builds the wheel and source distribution once. It uploads the same
GitHub artifact to TestPyPI and then to PyPI. The workflow does not rebuild between indexes.

Published files and versions are immutable. Do not delete and reuse a published version. Resume a
failed downstream job with the existing workflow artifact when GitHub permits it. Use a new patch
version when a released artifact needs a change.

## Configure Trusted Publishing

Configure the two package indexes before you push the first tag. TestPyPI and PyPI use separate
accounts and separate publisher records.

| setting | TestPyPI value | PyPI value |
| --- | --- | --- |
| GitHub owner | `esbraun` | `esbraun` |
| repository | `cleverly-tmle` | `cleverly-tmle` |
| workflow | `publish.yml` | `publish.yml` |
| environment | `testpypi` | `pypi` |

Create matching GitHub environments named `testpypi` and `pypi`. Require manual approval for every
deployment through the `pypi` environment.

Register each environment as a Trusted Publisher on its package index. The workflow requests an
OpenID Connect identity only inside the two publishing jobs. It stores no package-index token.

For a new project, use each index's pending Trusted Publisher form. The first accepted upload
creates the project. The PyPA
[publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
explains the account-side setup and environment protection.
