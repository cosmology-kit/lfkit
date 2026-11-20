# Changelog

All notable changes to LFKit will be documented in this file.

## [Unreleased]

## [0.7.0] - 2026-11-19

### Added

- Shared numerical utilities for safe division and exponentiation.
- Dedicated photometric flux conversion module.
- Generic CSV loading utilities for local files and packaged resources.
- Regression tests for parameter broadcasting, interpolation, photometric
  conversions, and packaged correction grids.
- Downloader tests using native Astroquery and Astropy table objects.

### Changed

- Consolidated shared typing aliases and array validation.
- Unified parameter evaluation through a single broadcasting-aware evaluator.
- Simplified numerical integration and interpolation utilities.
- Separated numerical operations from scientific calculations.
- Separated photometric flux conversions from physical unit conversions.
- Moved Poggianti-specific table processing into the corrections module.
- Moved the Poggianti data downloader into its corresponding data package.
- Moved K-correction grid serialization into the K-correction grid module.
- Migrated bundled K-correction grids to JSON metadata without changing
  their numerical arrays.
- Updated internal imports and regenerated the API documentation.
- Migrated documentation cards to the current Sphinx Design syntax.

### Removed

- Redundant parameter evaluation and numerical integration helpers.
- Scientific data-specific functionality from the generic utilities package.
- Pickle-based serialization of K-correction grid metadata.

### Fixed

- Scalar and multidimensional interpolation behavior.
- Deterministic handling of duplicate interpolation coordinates.
- Inconsistent K-correction grid metadata keys.
- Astroquery table iteration in the Poggianti downloader.
- Overly broad exception handling in response directory discovery.
- Repository-wide lint violations.

### Breaking changes

- Several internal utility functions have been consolidated or relocated.
- Existing external K-correction grid files using pickle-based metadata
  must be migrated to the new serialization format.
