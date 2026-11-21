# Changelog

All notable changes to LFKit will be documented in this file.

## [Unreleased]

### Added

- Dedicated parameters package for luminosity function parameter evolution.
- Parameter evolution overview with scientific definitions and executable
  examples.
- Regression tests for array-valued evolution coefficients, custom model
  outputs, normalization constraints, and numerical overflow.
- Inverse luminosity-to-magnitude conversion with configurable reference
  magnitude and luminosity.
- Optional little-h scaling in distance-based magnitude conversions.
- Photometry regression tests for round-trip conversions, broadcasting,
  invalid inputs, and floating-point limits.
- Executable examples for luminosity, magnitude, and flux conversions.
- Backend-independent cosmology utilities supporting Astropy and PyCCL.
- Dedicated modules for cosmological distances, expansion history,
  differential comoving volumes, and lookback times.
- Numerical inversion of the lookback-time relation.
- Cosmology overview and backend usage documentation.
- Regression tests covering flat, curved, and evolving dark energy
  cosmologies with both supported backends.
- CI configurations with and without the optional PyCCL dependency.

### Changed

- Separated parameter evolution models from luminosity function models.
- Consolidated parameter evolution through the shared parameter evaluator.
- Standardized parameter evolution outputs as float64 NumPy arrays
  broadcast to the redshift input shape.
- Consolidated luminosity ratios and luminosity weights through a shared
  magnitude-based conversion.
- Unified cosmology-based and distance-based magnitude conversions.
- Standardized photometric conversion return types as float64 NumPy arrays.
- Used shared cosmology utilities for luminosity distance validation.
- Made Astropy an explicit required dependency.
- Made PyCCL an optional dependency through the `ccl` installation extra.
- Migrated photometric magnitude conversions and catalog completeness
  calculations to the shared cosmology utilities.
- Consolidated lookback-time calculations and redshift inversion.
- Changed the default cosmology backend for Poggianti evolution
  correction remapping from PyCCL to Astropy.
- Reused shared cosmology fixtures throughout the test suite.

### Fixed

- Missing validation of non-finite parameter evolution coefficients
  and evaluated parameter values.
- Inconsistent scalar and array-valued parameter broadcasting.
- Missing validation of registered custom parameter model outputs.
- Acceptance of negative luminosity function normalization in
  parameter evolution models.
- Inconsistent validation of reference magnitudes and luminosities.
- Inconsistent zero-distance behavior between magnitude conversion interfaces.
- Missing validation of the configurable photometric flux floor.
- Numerical overflow and underflow in photometric inverse variance calculations.
- Comoving volume calculations for cosmologies with nonzero curvature.
- Distance calculations that depended on the supplied redshift grid.
- Background calculations that previously assumed a restricted
  cosmological expansion history.
- Scalar and multidimensional cosmology input handling.

### Removed

- Redundant photometry-specific luminosity distance validator.
- Redundant cosmology constructors and background integration routines.
- Duplicate lookback-time inversion helpers.

### Breaking changes

- Moved `lfkit.luminosity_functions.parameter_models` to
  `lfkit.parameters.evolution`.
- Parameter evolution models now reject negative normalization,
  non-finite coefficients, non-finite outputs, and incompatible shapes.
- Parameter model outputs evaluated through the shared evaluator must
  be broadcastable to the redshift input shape.
- Broadcast parameter arrays may be read-only.
- Removed `lfkit.utils.validators.validate_luminosity_distance`.
- Distance-based magnitude conversions now accept zero distance and
  return the corresponding infinite magnitude limit.
- Non-finite photometric inverse variances are now masked to zero.
- Invalid flux floors and non-finite reference luminosities now raise
  `ValueError`.
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
