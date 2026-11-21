.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo
   :width: 50px

|lfkitlogo| Parameter evolution
===============================

Luminosity function parameters can depend on redshift and other physical
quantities. LFKit separates parameter evolution from luminosity function
evaluation so that these components can be developed and used independently.

The :mod:`lfkit.parameters.evolution` module currently provides models for
the redshift evolution of Schechter luminosity function parameters.


Density evolution
-----------------

The normalization of the luminosity function may remain constant,

.. math::

   \phi_\star(z) = \phi_\star,

or evolve according to

.. math::

   \phi_\star(z) = \phi_{0,\star} 10^{0.4 P z}.

The parameter :math:`P` controls the evolution of the normalization.
Positive values increase the normalization with redshift, while negative
values decrease it.

The available models are ``constant`` and ``linear_p``.


Luminosity evolution
--------------------

The characteristic absolute magnitude may remain constant,

.. math::

   M_\star(z) = M_\star,

or evolve linearly with redshift,

.. math::

   M_\star(z) = M_{0,\star} - Q(z-z_{\mathrm{ref}}).

Positive values of :math:`Q` correspond to a brighter characteristic
magnitude at higher redshift.

The default reference redshift is :math:`z_{\mathrm{ref}}=0.1`.

The available models are ``constant`` and ``linear_q``.


Faint-end slope evolution
-------------------------

The faint-end slope may remain constant,

.. math::

   \alpha(z) = \alpha,

or evolve linearly,

.. math::

   \alpha(z) = \alpha_0 + \alpha_1(z-z_{\mathrm{ref}}).

The default reference redshift is :math:`z_{\mathrm{ref}}=0.1`.

The available models are ``constant`` and ``linear``.


Parameter evaluation
--------------------

All three parameters can be evaluated together using
:func:`lfkit.parameters.evolution.evaluate_lf_parameters`.

.. testcode::

   import numpy as np

   from lfkit.parameters.evolution import evaluate_lf_parameters

   z = np.linspace(0.0, 2.0, 100)

   phi_star, m_star, alpha = evaluate_lf_parameters(
       z,
       phi_model="linear_p",
       phi_kwargs={"phi_0_star": 1e-3, "p": 0.7},
       m_star_model="linear_q",
       m_star_kwargs={"m_0_star": -20.5, "q": 0.8},
       alpha_model="constant",
       alpha_kwargs={"alpha": -1.2},
   )

   assert phi_star.shape == z.shape
   assert m_star.shape == z.shape
   assert alpha.shape == z.shape


Scalar and array-valued parameters
----------------------------------

Parameter evolution functions accept scalar and array-like redshift inputs.
Numerical coefficients may also be arrays, provided they can be broadcast
to the shape of the redshift input.

All evaluated parameters are returned as float64 NumPy arrays.
Broadcast parameter arrays may be read-only.

.. testcode::

   import numpy as np

   from lfkit.parameters.evolution import phi_star_linear_p

   z = np.broadcast_to(
       np.linspace(0.0, 2.0, 100)[:, None],
       (100, 3),
   )

   phi_star = phi_star_linear_p(
       z,
       phi_0_star=1e-3,
       p=[-0.5, 0.0, 0.5],
   )

   assert phi_star.shape == (100, 3)


Validation
----------

Redshift inputs, numerical coefficients, and evaluated parameter values
must be finite.

Luminosity function normalization must be non-negative.
Zero normalization is permitted.

Characteristic magnitudes, faint-end slopes, and evolution coefficients
may take negative values.

Numerical overflow producing non-finite parameter values raises an error.
Incompatible parameter shapes also raise an error.

These output validation and broadcasting requirements also apply to
custom models evaluated through ``evaluate_lf_parameters``.


Custom parameter models
-----------------------

Users can register their own parameter evolution functions using
``register_phi_star_model``, ``register_m_star_model``, and
``register_alpha_model``.

A custom model accepts redshift as its first argument, followed by any
required model-specific keyword arguments.

Its output must be broadcastable to the shape of the redshift input.

Available models can be inspected using
:func:`lfkit.parameters.evolution.available_lf_parameter_models`.
