# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.abspath('../../ahrs' if sys.platform.startswith('win') else '../..'))
from tools.versioning import get_version

# -- Project information -----------------------------------------------------
project = 'AHRS'
author = 'Mario Garcia'
copyright = f"2019-{datetime.now().year}, {author}"
release = get_version()

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon']

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_use_ivar = True            # List attributes with :ivar:

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# Explicitly assign master document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['build']

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_logo = "ahrs_logo.png"
html_favicon = "ahrs_icon.ico"
