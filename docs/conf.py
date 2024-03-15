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

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
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
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
    "hoverxref.extension",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "code_include.extension",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]
html_favicon = "./_static/images/gflownet-logo-32.png"

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
    "top_of_page_button": None,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/alexhernandezgarcia/gflownet",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
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
autoapi_template_dir = "_templates/autoapi"
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
myst_enable_extensions = ["colon_fence"]

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


# def skip_util_classes(app, what, name, obj, skip, options):
#     return any(
#         name.startswith(f"gflownet.{p}") for p in ["envs", "proxy", "policy", "utils"]
#     )


# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_util_classes)
