# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

from os import getenv

from viresclient import set_token

import swarmpal

# Warning: do not change the path here. To use autodoc, you need to install the
# package first.

# -- Project information -----------------------------------------------------

project = "SwarmPAL"
copyright = "2026, Swarm DISC"
author = "The SwarmPAL developers"

# The version is derived from git tags by hatch-vcs at install time and written
# into swarmpal._version (see the vcs build hook in pyproject.toml). We read it
# from the imported package rather than importlib.metadata: an editable install
# can leave the .dist-info metadata pinned to a stale version while _version.py
# is regenerated from the current git state, so metadata would show an older
# release. On Read the Docs this requires the full git history (see the
# post_checkout job in .readthedocs.yaml) so that the nearest tag is reachable.
release = swarmpal.__version__
# The short X.Y version
version = ".".join(release.split(".")[:2])


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
fast_docs = bool(getenv("FAST_DOCS"))
extensions = [
    "myst_nb",
    *([] if fast_docs else ["autoapi.extension"]),
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store", ".env"]

# -- Extra configurations ----------------------------------------------------

autoapi_dirs = ["../src/swarmpal"]
# Avoid documenting stray Jupyter checkpoint copies of modules
autoapi_ignore = ["*/.ipynb_checkpoints/*"]

# -- Notebook execution config -----------------------------------------------

# Increase allowed notebook run time
# 15 minutes max per cell
# (should change this to use per-cell metadata instead)
nb_execution_timeout = 900
# Fix execution of notebooks with different kernel names
nb_kernel_rgx_aliases = {".*": "python3"}
# Temporarily disable notebook execution while working on docs (default is "auto")
# nb_execution_mode = "off"
# On Read the Docs, only execute notebooks for real branch/tag builds. Pull
# request preview builds (VERSION_TYPE == "external") skip execution to stay
# fast and just render the committed notebook outputs. Notebooks are executed
# for real once a change is merged (e.g. to the staging branch).
if getenv("READTHEDOCS_VERSION_TYPE") == "external":
    nb_execution_mode = "off"
else:
    nb_execution_mode = "cache"
# Errors in notebooks will only trigger a warning
# Use sphinx option "--fail-on-warning" to make the build report as failure
# This allows readthedocs to report failure in CI, while still displaying the docs
nb_execution_allow_errors = False
nb_execution_raise_on_error = False
# Merge consecutive stream outputs (e.g. chunked stdout from %%bash)
# so they render as a single <pre> block instead of being split mid-word
nb_merge_streams = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

html_title = f"{project} {version}"

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
