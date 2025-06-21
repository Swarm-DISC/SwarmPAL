# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

from os import getenv

from viresclient import set_token

# Warning: do not change the path here. To use autodoc, you need to install the
# package first.

# -- Project information -----------------------------------------------------

project = "SwarmPAL"
copyright = "2023, The SwarmPAL developers"
author = "The SwarmPAL developers"


# -- VirES access config -----------------------------------------------------
# This environment variable is set in readthedocs so that the docs build there
# is able to access VirES to run the notebook code used in the docs
token = getenv("VIRES_TOKEN")
if token:
    set_token(url="https://vires.services/ows", token=token, set_default=True)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # "myst_parser",  # broke when enabling this with myst_nb
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx.ext.viewcode",
    "sphinx_click",
    "sphinx_design",
    "sphinx_tabs.tabs",
]

myst_enable_extensions = [
    "colon_fence",
]

# Increase allowed notebook run time
nb_execution_timeout = 300
# Fix execution of notebooks with different kernel names
nb_kernel_rgx_aliases = {".*": "python3"}
# Temporarily disable notebook execution while working on docs (default is "auto")
# nb_execution_mode = "off"
# Errors in notebooks will only trigger a warning
# Use sphinx option "--fail-on-warning" to make the build report as failure
# This allows readthedocs to report failure in CI, while still displaying the docs
nb_execution_allow_errors = False
nb_execution_raise_on_error = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_title = f"{project}"

html_baseurl = "https://swarmpal.readthedocs.io/en/latest/"

html_theme_options = {
    "home_page_in_toc": True,
    "repository_url": "https://github.com/Swarm-DISC/SwarmPAL",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path: list[str] = []

# Fix https://github.com/executablebooks/sphinx-book-theme/issues/105
html_sourcelink_suffix = ""
