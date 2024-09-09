Installation
============

If you already have Python, you can install AHRS with:

.. code-block:: console

    pip install ahrs

This will download and install the latest stable release of ``ahrs`` available
in the `Python Package Index <https://pypi.org/>`_.

AHRS works with Python 3.6 or later, and depends on `NumPy <https://numpy.org/>`_.
Intalling with ``pip`` will automatically download it if it's not present in
your workspace.

A second, and more recommended, option is to download the *latest* available
version in the git repository and install it manually:

.. code-block:: console

    git clone https://github.com/Mayitzin/ahrs.git
    cd ahrs/
    python pip install .

To install as editable do this:

.. code-block:: console

    git clone https://github.com/Mayitzin/ahrs.git
    cd ahrs/
    python pip install -e .
 
To install specific requirements, do this:

.. code-block:: console

   python pip install .[dev]
   python pip install .[docs]

This will get you the latest changes of the package, so you can get an updated
version.

Building the Documentation
--------------------------

To build this documentation you first need to have `Sphinx
<https://www.sphinx-doc.org/en/master/>`_ and the `Pydata Sphinx Theme
<https://pydata-sphinx-theme.readthedocs.io/en/stable/index.html>`_.

.. code-block:: console
    
    python pip install .[docs]
    cd docs/
    make html

You can then build the documentation by running ``make html`` from the
``docs/`` folder to build the HTML documentation in the current folder. Run
``make`` to get a list of all available output formats.
