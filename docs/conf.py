"""Sphinx configuration for the cleverly documentation site."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cleverly import __version__  # noqa: E402

project = "cleverly"
author = "cleverly contributors"
copyright = "2026, cleverly contributors"
version = __version__
release = __version__

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

root_doc = "index"
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "api/generated/*.md", "Thumbs.db", ".DS_Store"]

# Notebooks are executed deliberately before review and commit their outputs.  Documentation
# builds remain deterministic, offline, and quick: they render those stored outputs without
# downloading data or refitting estimators.
nb_execution_mode = "off"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "cleverly: the Python toolbox for TMLE"
html_baseurl = "https://esbraun.github.io/cleverly-tmle/"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/esbraun/cleverly-tmle",
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "esbraun",
    "github_repo": "cleverly-tmle",
    "github_version": "main",
    "doc_path": "docs",
}

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d+\]: | {2,5}\.\.\.: "
copybutton_prompt_is_regexp = True
