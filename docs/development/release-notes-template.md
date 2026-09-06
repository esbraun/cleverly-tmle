# Release notes template

Use this template for each GitHub release. PyPI links to the same release history through the
`Changelog` project link in package metadata.

## What the template optimizes for

A release note summarizes merged pull requests for a package user. It does not restate the pull
request discussion or advertise the project.

Write each bullet from the merged diff and its reviewed pull request. Start with the behavior that
a user can observe. Name the affected API, supported method, dependency, or installation path.
Link the pull request at the end.

Use these rules during review.

| rule | application |
| --- | --- |
| report facts | State what changed and the condition under which it matters. |
| use specific subjects | Name an estimator, method, function, or package extra. |
| keep one outcome per bullet | Split unrelated outcomes, even when one pull request contains both. |
| identify required action | Put migration work in `Upgrade notes`, before the change list. |
| preserve scientific limits | State the estimand and conditions for a new statistical claim. |
| omit process detail | Exclude refactors, test rewrites, and review history unless they alter shipped behavior. |
| remove filler | Do not use claims such as “powerful”, “effortless”, “robust”, or “exciting”. |
| keep attribution | End each bullet with its pull request link. Name new contributors once. |

Do not ask a text generator to infer release claims from a commit list. A maintainer checks every
bullet against the merged change. The pull request link lets a reader inspect the evidence.

## Copy this template

Delete every comment and empty section before publication.

```markdown
# cleverly 0.1.N

<!-- Optional. Use one factual sentence when the release has a coherent theme. -->

## Upgrade notes

<!-- Keep this section only when a user must act. Put the required action first. -->

- Replace `old_call()` with `new_call()`. `old_call()` now ... ([#PR](PR-URL))

## Added

- `public_name`: Adds [behavior] for [use case] under [conditions]. ([#PR](PR-URL))

## Changed

- `public_name`: Changes [behavior] when [condition]. Existing [behavior] remains unchanged. ([#PR](PR-URL))

## Fixed

- `public_name`: Corrects [behavior] when [condition]. Earlier versions [effect]. ([#PR](PR-URL))

## Deprecated

- `public_name`: Deprecates [API]. Use [replacement] before removal in [version]. ([#PR](PR-URL))

## Packaging and compatibility

- Supports [version or platform] and requires [dependency constraint]. ([#PR](PR-URL))

## Contributors

- Name ([#PR](PR-URL))

**Full diff:** [`vPREVIOUS...v0.1.N`](COMPARE-URL)
```

Use `Removed` after the alpha series when a published deprecation reaches its stated removal
version. During the alpha series, describe an incompatible change in `Upgrade notes` even when no
deprecation preceded it.

## Prepare the notes

1. List the pull requests merged after the prior release tag.
2. Exclude a pull request that changes no installed behavior or user documentation.
3. Assign each remaining user outcome to one template section.
4. Write the outcome from the diff, then append the pull request link.
5. Check every compatibility statement and statistical claim against its source evidence.
6. Remove empty sections, placeholders, repeated claims, and promotional language.
7. Ask the release pull request reviewer to compare the notes with the full diff.
8. Publish the reviewed text on the GitHub release after the tag workflow succeeds.

## Basis for this structure

The large scientific Python libraries use stable headings instead of a narrative generated from
commit messages. Their exact taxonomies differ, but each separates user-facing changes by kind.

| source | practice used here |
| --- | --- |
| [NumPy release notes](https://numpy.org/doc/stable/release.html) | Keep versioned notes and separate new work, changes, compatibility, and deprecations. |
| [SciPy release notes](https://docs.scipy.org/doc/scipy/release.html) | Give highlights before categorized changes, then link the underlying work. |
| [pandas release notes](https://pandas.pydata.org/docs/whatsnew/index.html) | Put incompatible changes and deprecations outside routine bug fixes. |
| [scikit-learn release history](https://scikit-learn.org/stable/whats_new.html) | Name the affected public area and credit the contributor with the change. |
| [GitHub release-note documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes) | Use pull requests as traceable inputs, but review the final categorization and wording. |
| [PyPA core metadata](https://packaging.python.org/en/latest/specifications/core-metadata/#project-url-multiple-use) | Publish the release history as a labeled project URL in package metadata. |

This project keeps the short category set because its releases contain fewer pull requests. The
maintainer adds a narrower heading only when two or more bullets need it.
