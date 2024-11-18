# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os

project = 'Llettuce'
copyright = '2024, Reza Omidvar, James Mitchell-White, Grazziela Figueredo, Philip Quinlan'
author = 'Reza Omidvar, James Mitchell-White, Grazziela Figueredo, Philip Quinlan'
release = '0.1'

# Add or modify these lines
html_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs'))
html_outdir = html_build_dir
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'autoapi.extension',
    'sphinxcontrib.mermaid'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autoapi_dirs = ['../Carrot-Assistant/']
autoapi_keep_files = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
