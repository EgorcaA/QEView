# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))

project = 'QEView'
copyright = '2025, Egor M. Agapov'
author = 'Egor M. Agapov'
release = '1.0.6'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.todo", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.autodoc",   
              'sphinx.ext.autosummary', 
              "sphinx.ext.napoleon",
              'nbsphinx',  # Enables Jupyter Notebook support
              'sphinx.ext.mathjax',  # Optional: for rendering LaTeX equations
              ]

# Napoleon settings (for NumPy/Google docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# The master toctree document.
master_doc = 'contents'

templates_path = ['_templates']
# exclude_patterns = []
# source_suffix = ['.rst', '.md', '.ipynb']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
nbsphinx_execute = 'never'  # Prevent execution of code cells

# nbsphinx_allow_errors = True