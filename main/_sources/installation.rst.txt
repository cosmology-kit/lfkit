.. |lfkitlogo| image:: /_static/logos/lfkit_logo.png
   :alt: LFKit logo black
   :width: 50px


|lfkitlogo| Installation
========================

LFKit can be installed from PyPI. The PyPI distribution name is ``py-lfkit``::

   pip3 install py-lfkit

LFKit can additionally be installed from source.
To create an editable installation, clone the repository and run::

   git clone https://github.com/cosmology-kit/lfkit.git
   cd lfkit
   pip install -e .

Alternatively, install directly from GitHub with ``pip``::

   pip install git+https://github.com/cosmology-kit/lfkit.git


Development installation
------------------------

For development work, clone the repository and install LFKit in editable mode
with the development dependencies::

   git clone https://github.com/cosmology-kit/lfkit.git
   cd lfkit
   pip install -e ".[dev]"

This allows local changes to the source code to be picked up without
reinstalling the package.

To run the full test suite with both supported cosmology backends, install::

   pip install -e ".[dev,ccl]"

To build the documentation, install::

   pip install -e ".[docs,viz,ccl]"


Optional dependencies
---------------------

LFKit uses Astropy for its default background cosmology calculations.
PyCCL is supported as an optional cosmology backend and is not required
for a standard installation.

To install LFKit with PyCCL support, run::

   pip install "py-lfkit[ccl]"

For plotting and visualization utilities, install::

   pip install "py-lfkit[viz]"

Multiple optional dependency groups can be installed together::

   pip install "py-lfkit[ccl,viz]"
