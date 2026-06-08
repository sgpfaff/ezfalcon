# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

import datetime

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags
from tambora import __version__

release = __version__
version = __version__.split("+")[0]  # short version for display

project = "tambora"
author = "Gabriel Pfaffman"
copyright = f"{datetime.datetime.now().year}, {author}"  # noqa: A001

html_title = project
html_short_title = project

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx_automodapi.smart_resolver",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

# -- MyST / MyST-NB ----------------------------------------------------------
# myst_nb activates myst_parser; do NOT also list "myst_parser" above.
myst_enable_extensions = [
    "dollarmath",   # $...$ / $$...$$ math (used in the user guide)
    "colon_fence",  # ::: fences as an alternative to ```{directive}
]

# Notebooks are never executed at build time. The User Guide pages are .ipynb
# notebooks committed *with their outputs* (re-run them locally whenever you
# want to refresh the docs), and the Examples render committed outputs too, so
# the build just renders what's stored. This keeps RTD builds fast and means
# the build needs no science dependencies to run cells.
nb_execution_mode = "off"

# Add any paths that contain templates here, relative to this directory.
# templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store",
    # Internal scratch notes — not a docs page (now that .md is a source suffix).
    "tests_and_docs_to_write.md",
    # All example notebooks — docs use .rst pages + _figures/ instead
    "examples/*.ipynb",
    "examples/EX1_Multicomponent_Shell.rst",
    "examples/EX2_Disk.rst",
    "examples/EX4_satellite_script.py",
    "examples/EX5_GC_Satellite_script.py",
    "tutorials/**",
]

# The suffix(es) of source filenames.
# myst_nb parses .md (MyST Markdown, incl. text-based notebooks with
# {code-cell} blocks) and .ipynb; .rst stays on the reStructuredText parser.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# Treat everything in single ` as a Python reference.
default_role = 'py:obj'

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sgpfaff/tambora",
            "icon": "fa-brands fa-github",
        },
    ],
    # Navbar carries the top-level sections; the left sidebar shows only the
    # pages *within* the current section (show_nav_level=1) so the two don't
    # duplicate each other.
    "navbar_align": "left",
    "show_nav_level": 1,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "show_toc_level": 2,
}

# Single-page sections have nothing to put in a left sidebar (it would just
# echo the navbar), so drop it for a cleaner, wider layout there.
html_sidebars = {
    "installation": [],
    "quickstart": [],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# Plotly requires its JS bundle to render application/vnd.plotly.v1+json outputs.
# The CDN bundle is injected into every page so interactive Plotly charts in
# notebooks work without needing requirejs.
html_js_files = [
    "https://cdn.plot.ly/plotly-2.35.2.min.js",
]

# By default, when rendering docstrings for classes, sphinx.ext.autodoc will
# make docs with the class-level docstring and the class-method docstrings,
# but not the __init__ docstring, which often contains the parameters to
# class constructors across the scientific Python ecosystem. The option below
# will append the __init__ docstring to the class-level docstring when rendering
# the docs. For more options, see:
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autoclass_content
autoclass_content = "both"

# -- Other options ----------------------------------------------------------

myst_heading_anchors = 3  # add anchors to all h1, h2, h3 headings for easy linking