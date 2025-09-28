# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
from pathlib import Path

# More reliable path setup for ReadTheDocs
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "gflownet"
copyright = "2024, Alex Hernandez-Garcia"
author = "Alex Hernandez-Garcia, Nikita Saxena, Alexandra Volokhova, Michał Koziarski, Divya Sharma, Joseph D Viviano, Pierre Luc Carrier, Victor Schmidt and others."


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_math_dollar",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.todo",
    "sphinx_markdown_tables",
    "sphinx_autodoc_typehints",
    "myst_nb",
    "hoverxref.extension",
    "autoapi.extension",
    "sphinxext.opengraph",
    "code_include.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source parsers and suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
html_favicon = "./_static/images/gflownet-logo.png"
html_logo = "./figures/reward_landscape.png"
# -----------------------------
# -----  Plugins configs  -----
# -----------------------------

# Napoleon
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#directive-todo
todo_include_todos = True

# Furo theme
# https://pradyunsg.me/furo/customisation/
html_theme_options = {
    "logo_only": True,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
}

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable", None),
    "omegaconf": ("https://omegaconf.readthedocs.io/en/latest", None),
}

# sphinx.ext.autodoc & autoapi.extension
# https://autoapi.readthedocs.io/
autodoc_typehints = "description"
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "gflownet")]
autoapi_member_order = "bysource"
# autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "init"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",
    "special-members",
]
autoapi_keep_files = False

# sphinx_math_dollar
# Note: CHTML is the only output format that works with \mathcal{}
mathjax_path = "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_CHTML"
mathjax3_config = {
    "tex": {
        "inlineMath": [
            ["$", "$"],
            ["\\(", "\\)"],
        ],
        "processEscapes": True,
    },
    "jax": ["input/TeX", "output/CommonHTML", "output/HTML-CSS"],
}

# sphinx_autodoc_typehints
# https://github.com/tox-dev/sphinx-autodoc-typehints
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
typehints_defaults = "comma"

# MyST
# https://myst-parser.readthedocs.io/en/latest/intro.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# Hover X Ref
# https://sphinx-hoverxref.readthedocs.io/en/latest/index.html
hoverxref_auto_ref = True
hoverxref_mathjax = True

# Open Graph

ogp_site_url = "https://gflownet.readthedocs.io/en/latest/"
ogp_social_cards = {
    "enable": True,
    "image": "./_static/images/gflownet-logo.png",
}

# Jupyter notebook execution
nb_execution_mode = "off"

# Default role
default_role = "code"

# def skip_util_classes(app, what, name, obj, skip, options):
#     return any(
#         name.startswith(f"gflownet.{p}") for p in ["envs", "proxy", "policy", "utils"]
#     )


# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_util_classes)
