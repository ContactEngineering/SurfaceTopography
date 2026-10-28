# Contributing

## Development setup

Install the package in editable mode as described under
[Development installation](installation.md#development-installation). The
extension module is rebuilt automatically when C++ sources in `cpp/` change.
Run the [tests](testing.md) before opening a pull request.

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/). CI runs `flake8` on
  `SurfaceTopography` and `test` with a maximum line length of 120 characters
  (`setup.cfg`).
- Docstrings follow the [numpydoc](https://numpydoc.readthedocs.io/) standard.
  State units and defaults in the parameter descriptions, e.g.
  `(Units: length)` and `(Default: None)`, and write equations with
  `:math:` and `.. math::`.
- Name parameters after what they are, not after a symbol: use
  `long_cutoff`, not `lambda`.
- Functions that analyze or transform topographies are registered on the
  topography classes (see [Topographies](topographies.md#pipelines)); a new
  analysis belongs in the module of its family (e.g. `Uniform/`, `Nonuniform/`,
  `Generic/`).
- Mathematical notation in docstrings and in this documentation follows the
  symbols already in use (e.g. $h$ for heights, $L$ for physical sizes,
  $C^\text{1D}$ and $C^\text{2D}$ for power spectra, $A(r)$ for the
  autocorrelation function).

## Branches

Develop every feature in its own branch, named `<YY>_<description>` with the
two-digit year of creation, e.g. `48_hyperdimensional_scans` for work started
in 2048. Pull requests target `main`.

## Commits

Prefix commit messages and pull request titles with the type of change:

| Prefix | Change |
|--------|--------|
| `API` | changes to the user-exposed API |
| `BUG` | bug fix |
| `BUILD` | build system |
| `CI` | continuous integration |
| `DOC` | documentation (docstrings or pages, not only typos) |
| `ENH` | enhancement, e.g. a new feature |
| `MAINT` | maintenance, e.g. typos or refactoring without change of behavior |
| `PERF` | speed or memory |
| `TST` | tests |
| `WIP` | work in progress |

- Use `WIP` while a feature is incomplete and `ENH` on the commit that makes it
  ready.
- Use `TST` for changes that only affect tests; a bug fix together with its
  test is `BUG`.
- Use `DOC` only for commits that do not change code.

The [changelog](changelog.md) lists user-visible changes (`BUG`, `API`, `ENH`,
`PERF`). Add an entry to `CHANGELOG.md` with your pull request; the comment at
the top of that file describes its style.

## Authors

Add yourself to `AUTHORS` with the email address you use for your commits.
The copyright statements in the file headers are generated from the commit log
by `maintenance/update_license_headers.sh`.

## Documentation

The documentation is written in Markdown in `docs/` and built with
[MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme; equations are
rendered by MathJax (`$...$` and `$$...$$`). Preview it with

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

and check that `mkdocs build --strict` passes. CI builds the site for every
pull request and publishes it to GitHub Pages from `main`.
