# Changelog

Notable changes to QuditsOnQubits are recorded here, using the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. The 0.1.0 and
0.1.1 entries were reconstructed from the tagged Git history.

## [Unreleased]

## [0.1.2] - 2026-09-19

### Added

- Unified encoding benchmark API and `qoq-benchmark` CLI, with LocalSU2 and
  SchmidtTheta candidate families, backend-specific compilation, validation,
  artifact provenance, and Pareto analysis.
- Portable encoding benchmark notebook and saved theta gate data for an offline
  example, plus source/wheel installation verification.
- Contributor Code of Conduct, security policy, issue templates, and pull request
  template.
- English results summary of the archived IBM and IQM hardware campaign in
  `benchmarks/bell_20260907/README.md`.
- `version` and `date-released` metadata in `CITATION.cff`.
- Community policies, changelog, and GitHub issue/PR templates in the source
  distribution, with required-file checks in the distribution verifier.

### Changed

- Theta continuation, reassessment, and hardware reports now use English headings,
  table labels, and plot text. Boolean cells use `yes`/`no` instead of `tak`/`nie`,
  and `gallery.html` declares `lang="en"` instead of `lang="pl"`.
- `benchmark_encoding_bases.py` displays missing values as `missing` instead of
  `brak` and emits English `error_message` diagnostics.
- Translated `docs/theta_benchmark.md` and `docs/theta-hardware-layer.md` to English.
- Translated Polish code comments in `notebooks/CZ3_bqckit_optimalization.ipynb`
  and `notebooks/f3_optimalization.ipynb`, plus the working IQM AME and ZNE
  notebooks, to English.
- Links to archived Polish documents now carry a `(Polish)` label.
- Corrected the duplicated `Copyright` word in `LICENSE`.
- Distribution verification selects artifacts with version-independent globs.
  Local contribution instructions use a fresh output directory; CI builds in a
  clean checkout.

### Fixed

- Portable theta replay and propagation of the requested transpiler optimization
  level in the direct-basis benchmark command.
- Removed user-specific absolute paths from published Bell report metadata and
  the historical validation log; case-insensitive checks cover JSON and text
  artifacts.
- Corrected the README installation anchor in
  `docs/two_qutrit_bell_vertical_slice.md` from `#install-from-source` to `#install`.

## [0.1.1] - 2026-09-14

### Fixed

- Aligned usage-guide contract tests with the current documentation and made
  failures identify missing runner-contract terms.

## [0.1.0] - 2026-09-14

### Added

- Initial Apache-2.0 Python distribution for Python 3.11–3.13, including packaged
  QPY gate data.
- Two-qutrit Bell workflow and `qoq-two-qutrit-bell` CLI, with persisted run
  manifests for reproducible execution.
- Published IBM and IQM Bell benchmark data with checksummed offline replay in
  `benchmarks/bell_20260907/`.
- Direct-basis gate optimization and independent local-setting raw Bell
  experiments.

[Unreleased]: https://github.com/slysek/QuditsOnQubits/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/slysek/QuditsOnQubits/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/slysek/QuditsOnQubits/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/slysek/QuditsOnQubits/releases/tag/v0.1.0
