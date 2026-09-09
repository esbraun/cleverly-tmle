# Canonical comparison studies

This directory holds the registered comparison studies. Each study runs a reference R package in
a pinned container and compares its numbers against `cleverly`.

## License

**This directory is under the GNU General Public License v3.0. See [LICENSE](LICENSE).**

The rest of the project is under the MIT License. See the [LICENSE](../../LICENSE) at the
repository root.

The reason is the R runner scripts. A runner attaches the namespace of a reference package and
calls it in the same process. Several of those packages carry a copyleft license. The `lmtp`
package declares AGPL-3 in its `DESCRIPTION`, which is the strongest case here. The GPL applies
to this directory so that no reader has to judge that question.

## What this means for an installed package

Nothing in this directory reaches an installed copy of `cleverly`. Two build settings enforce
that:

| target | setting | effect |
| --- | --- | --- |
| wheel | `packages = ["src/cleverly"]` | ships the Python package alone |
| sdist | `exclude = ["/tests"]` | drops the whole test tree |

Both settings live in `pyproject.toml`. So a default install carries MIT code only.

The shipped Python package contains no R code. It does not import, link to, or bundle any R
package. Each reference package runs as a separate process inside a container. It reads sample
data that `cleverly` generated and writes estimates back as CSV. The committed artifacts in this
directory are those estimates.

## Regenerating a study

Read [method benchmarking](../../docs/development/method-benchmarking.md) first. Each study
directory carries its own `README.md` with the command that regenerates it.
