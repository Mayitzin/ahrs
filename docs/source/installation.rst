Installation
============

If you already have Python, you can install AHRS with:

.. code-block:: console

    pip install ahrs

This will download and install the latest stable release of ``ahrs`` available
in the `Python Package Index <https://pypi.org/>`_.

AHRS works with Python 3.6 or later, and depends on `NumPy <https://numpy.org/>`_
and `SciPy <https://www.scipy.org/>`_. Intalling with ``pip`` will automatically
download them if they are not present in your workspace.

A second option is to download the *latest* available version in the repository
and install it manually:

.. code-block:: console

    git clone https://github.com/Mayitzin/ahrs.git
    cd ahrs/
    python setup.py install

This is specially recommended if you want to have the newest version of the
package.

Building the Documentation
--------------------------

To build this documentation you first need to have Sphinx and the readthedocs
theme.

.. code-block:: console

    cd docs/
    pip install -r requirements.txt

You can then build the documentation by running ``make html`` from the
``docs/`` folder to build the HTML documentation in the current folder. Run
``make`` to get a list of all available output formats.
