# ruff: noqa: E501

"""Sphinx configuration file."""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import importlib.metadata
import os
import sys


sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "torch-sim-atomistic"
copyright = "2025, Radical AI"  # noqa: A001
author = "Abhijeet Gangan, Orion Cohen, Janosh Riebesell"

# The short X.Y version
version = importlib.metadata.version(project)
# The full version, including alpha/beta/rc tags
release = importlib.metadata.version(project)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = (
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxcontrib.autodoc_pydantic",
    "numpydoc",
    "nbsphinx",
    "sphinx_design",
    "sphinx_copybutton",
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Thumbs.db", ".DS_Store", "test*.py"]

myst_heading_anchors = 2  # enable headings as link targets
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
    "html_admonition",
    "html_image",
]

autodoc_mock_imports = [
    "fairchem",
    "mace",
    "mattersim",
    "metatensor",
    "orb",
    "sevennet",
    "graphpes",
]

# use type hints
autodoc_typehints = "description"
# autoclass_content = "both"
# autodoc_member_order = "bysource"

# better napoleon support
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_ivar = True

# The suffix(es) of source filenames.
source_suffix = {".rst": "restructuredtext", ".md": "restructuredtext"}

mathjax3_config = {
    "tex": {
        "macros": {
            "N": "\\mathbb{N}",
            "floor": ["\\lfloor#1\\rfloor", 1],
            "bmat": ["\\left[\\begin{array}"],
            "emat": ["\\end{array}\\right]"],
        }
    }
}
latex_elements = {
    "preamble": r"""\newcommand\N{\mathbb{N}}
\newcommand\floor[1]{\lfloor#1\rfloor}
\newcommand{\bmat}{\left[\begin{array}}
\newcommand{\emat}{\end{array}\right]}
"""
}
language = "en"
# html_extra_path = ["images/badge.svg"]
html_static_path = ["_static"]
html_css_files = ["custom.css", "github.css"]
suppress_warnings = ["etoc.toctree"]

# autodoc options
autosummary_imported_members = False
autodoc_preserve_defaults = True
autoclass_content = "class"
autodoc_member_order = "bysource"

python_use_unqualified_type_names = True

# don't overwrite summary but generate them if they don't exist
autosummary_generate = True
autosummary_generate_overwrite = True

# numpydoc options
numpydoc_class_members_toctree = False
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False
numpydoc_xref_param_type = True

# sphinx-panels shouldn't add bootstrap css as the pydata-sphinx-theme already loads it
panels_add_bootstrap_css = False

# control pydantic model docs
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_config = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_member_order = "bysource"
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_show_config = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_member_order = "bysource"
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = "furo"
html_theme_options = {
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Radical-AI/torch-sim",
            "html": """
<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
</svg>
            """,
            "class": "",
        },
    ],
    "source_repository": "https://github.com/Radical-AI/torch-sim/",
    "source_branch": "main",
    "source_directory": "docs/",
}


# hide sphinx footer
html_show_sphinx = False
html_show_sourcelink = False
html_title = "torch-sim"

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.10", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymatgen": ("https://pymatgen.org/", None),
}
