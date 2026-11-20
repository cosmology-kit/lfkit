.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Cosmology
=====================

LFKit provides background cosmology utilities for connecting galaxy
luminosity functions, observed photometry, survey volumes, and
redshift-dependent quantities.

These utilities support Astropy and PyCCL cosmology objects through a
common functional interface. Astropy is a required dependency, while
PyCCL is optional.

LFKit delegates background cosmology calculations to the supplied
cosmology object rather than implementing an independent cosmological
solver.


Cosmological distances
----------------------

The radial comoving distance is

.. math::

   \chi(z) = c \int_0^z \frac{\mathrm{d}z'}{H(z')}.

The transverse comoving distance, :math:`D_{\mathrm{M}}(z)`, accounts
for spatial curvature.

In a spatially flat universe,

.. math::

   D_{\mathrm{M}}(z) = \chi(z).

The luminosity distance is

.. math::

   D_{\mathrm{L}}(z) = (1+z)D_{\mathrm{M}}(z).

The distance modulus is

.. math::

   \mu(z) =
   5\log_{10}\left(\frac{D_{\mathrm{L}}(z)}{\mathrm{Mpc}}\right)
   + 25.

When an explicit dimensionless Hubble parameter :math:`h` is supplied
to the distance modulus helper, the distance inside the logarithm is
scaled by :math:`h`.

This convention is useful when working with luminosity functions
defined in terms of :math:`M-5\log_{10}h`.

See :mod:`lfkit.cosmology.distances` for the corresponding functions.


Expansion history
-----------------

The dimensionless expansion rate is defined as

.. math::

   E(z) = \frac{H(z)}{H_0},

where

.. math::

   H_0 = 100h\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1}.

LFKit evaluates the expansion history using the supplied cosmology
object, including its matter, radiation, curvature, and dark energy
components as configured by the user.

See :mod:`lfkit.cosmology.background`.


Comoving volume
---------------

For a survey covering a fraction :math:`f_{\mathrm{sky}}` of the sky,
the differential comoving volume is

.. math::

   \frac{\mathrm{d}V_{\mathrm{c}}}{\mathrm{d}z}
   =
   4\pi f_{\mathrm{sky}}
   \frac{cD_{\mathrm{M}}^2(z)}{H(z)}.

The transverse comoving distance ensures that the volume calculation
also applies to cosmologies with nonzero spatial curvature.

For a survey covering an area :math:`A` in square degrees,

.. math::

   f_{\mathrm{sky}} =
   \frac{A}{41252.96\,\mathrm{deg}^{2}}.

Combining the differential volume with a comoving galaxy number
density gives the corresponding redshift distribution in number counts:

.. math::

   \frac{\mathrm{d}N}{\mathrm{d}z}
   =
   n_{\mathrm{g}}(z)
   \frac{\mathrm{d}V_{\mathrm{c}}}{\mathrm{d}z}.

Here, :math:`n_{\mathrm{g}}(z)` may be obtained by integrating a
luminosity function over the magnitude range relevant to a survey.


Cosmological times
------------------

The lookback time is

.. math::

   t_{\mathrm{lb}}(z)
   =
   \int_0^z
   \frac{\mathrm{d}z'}{(1+z')H(z')}.

LFKit also provides numerical inversion of the lookback-time relation,

.. math::

   z = z(t_{\mathrm{lb}}),

using interpolation over a monotonically increasing lookback-time grid.

This is useful when translating galaxy evolution models between
different cosmological backgrounds.

In particular, LFKit uses this functionality when remapping the
Poggianti evolution correction tables.

See :mod:`lfkit.cosmology.times`.


Units and conventions
---------------------

All public cosmology functions accept redshift rather than scale factor.
Conversions to scale factor are handled internally when required by
the cosmology backend.

.. list-table::
   :header-rows: 1
   :widths: 65 35

   * - Quantity
     - Units
   * - Redshift
     - Dimensionless
   * - Dimensionless Hubble parameter
     - Dimensionless
   * - Expansion rate
     - Dimensionless
   * - Hubble parameter
     - km/s/Mpc
   * - Comoving distances
     - Mpc
   * - Luminosity distance
     - Mpc
   * - Distance modulus
     - Magnitudes
   * - Differential comoving volume
     - Mpc cubed per unit redshift
   * - Lookback time
     - Gyr

Distances and volumes are not expressed in little-h-scaled units.
The distance modulus helper supports explicit h scaling when needed.


Using Astropy
-------------

Astropy cosmology objects can be passed directly to LFKit.

.. code-block:: python

   import numpy as np

   from astropy.cosmology import FlatLambdaCDM

   from lfkit.cosmology.background import (
       differential_comoving_volume,
       expansion_rate,
   )
   from lfkit.cosmology.distances import luminosity_distance_mpc
   from lfkit.cosmology.times import (
       lookback_time_gyr,
       z_from_lookback_time,
   )

   cosmo = FlatLambdaCDM(
       H0=70.0,
       Om0=0.3,
       Ob0=0.05,
   )

   z = np.linspace(0.0, 2.0, 200)

   distance = luminosity_distance_mpc(cosmo, z)
   expansion = expansion_rate(cosmo, z)
   volume = differential_comoving_volume(cosmo, z, frac_sky=0.3)
   time = lookback_time_gyr(cosmo, z)
   recovered_z = z_from_lookback_time(cosmo, time)


Using PyCCL
-----------

PyCCL is supported through the same interface.

Install the optional dependency as described in :doc:`../installation`.

.. code-block:: python

   import numpy as np
   import pyccl as ccl

   from lfkit.cosmology.background import differential_comoving_volume
   from lfkit.cosmology.distances import luminosity_distance_mpc
   from lfkit.cosmology.times import lookback_time_gyr

   cosmo = ccl.Cosmology(
       Omega_c=0.25,
       Omega_b=0.05,
       h=0.7,
       sigma8=0.8,
       n_s=0.96,
   )

   z = np.linspace(0.0, 2.0, 200)

   distance = luminosity_distance_mpc(cosmo, z)
   volume = differential_comoving_volume(cosmo, z, frac_sky=0.3)
   time = lookback_time_gyr(cosmo, z)

The same LFKit functions can be used with either backend.
Numerical agreement between backends depends on matching their
cosmological parameters, physical assumptions, and numerical settings.
