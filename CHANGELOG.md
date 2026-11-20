# Changelog

All notable changes to LFKit will be documented in this file.

## [Unreleased]

### Added

- Backend-independent cosmology utilities supporting Astropy and PyCCL.
- Dedicated modules for cosmological distances, expansion history,
  differential comoving volumes, and lookback times.
- Numerical inversion of the lookback-time relation.
- Cosmology overview and backend usage documentation.
- Regression tests covering flat, curved, and evolving dark energy
  cosmologies with both supported backends.
- CI configurations with and without the optional PyCCL dependency.

### Changed

- Made Astropy an explicit required dependency.
- Made PyCCL an optional dependency through the `ccl` installation extra.
- Migrated photometric magnitude conversions and catalog completeness
  calculations to the shared cosmology utilities.
- Consolidated lookback-time calculations and redshift inversion.
- Changed the default cosmology backend for Poggianti evolution
  correction remapping from PyCCL to Astropy.
- Reused shared cosmology fixtures throughout the test suite.

### Fixed

- Comoving volume calculations for cosmologies with nonzero curvature.
- Distance calculations that depended on the supplied redshift grid.
- Background calculations that previously assumed a restricted
  cosmological expansion history.
- Scalar and multidimensional cosmology input handling.

### Removed

- Redundant cosmology constructors and background integration routines.
- Duplicate lookback-time inversion helpers.

### Breaking changes

- Removed the legacy `lfkit.cosmo` package and its cosmology constructor.
  Use Astropy or PyCCL cosmology objects with `lfkit.cosmology` instead.
- Replaced legacy background cosmology functions with the new
  backend-independent interface.
- Users requiring PyCCL must explicitly install the optional
  `py-lfkit[ccl]` dependency.
- Poggianti evolution correction remapping may exhibit small numerical
  differences due to the change in default cosmology backend.

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
